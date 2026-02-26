from db.tasks import (
    create_task_record,
    get_task_record,
    update_task_status,
    set_task_result,
    write_orchestration_log,
    get_task_logs,
)
from schemas.tasks import TaskCreateRequest
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from db.health import check_postgres, check_redis
from db.init_schema import init_schema
from db.inspect import (
    list_agent_instructions,
    list_communication_rules,
    list_public_tables,
)
from db.seed import seed_core_data

app = FastAPI(title="Ozonator Agents AA")


def require_admin_token(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    server_token = os.getenv("ADMIN_DEV_TOKEN")

    if not server_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_DEV_TOKEN не настроен на сервере",
        )

    if not x_admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не указан X-Admin-Token",
        )

    if x_admin_token != server_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный X-Admin-Token",
        )


@app.get("/")
def root():
    return {
        "service": "AA",
        "status": "ok",
        "message": "Агент-администратор запущен",
    }


@app.get("/health")
def health():
    return {
        "service": "AA",
        "status": "ok",
        "message": "Сервис работает",
    }


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, message = check_postgres(settings.database_url)

    payload = {
        "service": "AA",
        "component": "postgres",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, message = check_redis(settings.redis_url)

    payload = {
        "service": "AA",
        "component": "redis",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/health/all")
def health_all():
    settings = get_settings()

    db_ok, db_message = check_postgres(settings.database_url)
    redis_ok, redis_message = check_redis(settings.redis_url)
    app_ok = True
    all_ok = app_ok and db_ok and redis_ok

    payload = {
        "service": "AA",
        "status": "ok" if all_ok else "error",
        "components": {
            "app": {
                "status": "ok",
                "message": "Сервис работает",
            },
            "postgres": {
                "status": "ok" if db_ok else "error",
                "message": db_message,
            },
            "redis": {
                "status": "ok" if redis_ok else "error",
                "message": redis_message,
            },
        },
    }
    return JSONResponse(status_code=200 if all_ok else 503, content=payload)


@app.post("/admin/init-db", dependencies=[Depends(require_admin_token)])
def admin_init_db():
    settings = get_settings()
    ok, message = init_schema(settings.database_url)

    payload = {
        "service": "AA",
        "operation": "init_db_schema",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/tables", dependencies=[Depends(require_admin_token)])
def admin_tables():
    settings = get_settings()
    ok, tables, message = list_public_tables(settings.database_url)

    payload = {
        "service": "AA",
        "operation": "list_public_tables",
        "status": "ok" if ok else "error",
        "message": message,
        "tables": tables if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.post("/admin/seed-core", dependencies=[Depends(require_admin_token)])
def admin_seed_core():
    settings = get_settings()
    ok, details, message = seed_core_data(settings.database_url)

    payload = {
        "service": "AA",
        "operation": "seed_core_data",
        "status": "ok" if ok else "error",
        "message": message,
        "details": details if ok else {},
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/agent-instructions", dependencies=[Depends(require_admin_token)])
def admin_agent_instructions():
    settings = get_settings()
    ok, items, message = list_agent_instructions(settings.database_url)

    payload = {
        "service": "AA",
        "operation": "list_agent_instructions",
        "status": "ok" if ok else "error",
        "message": message,
        "items": items if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/communication-rules", dependencies=[Depends(require_admin_token)])
def admin_communication_rules():
    settings = get_settings()
    ok, items, message = list_communication_rules(settings.database_url)

    payload = {
        "service": "AA",
        "operation": "list_communication_rules",
        "status": "ok" if ok else "error",
        "message": message,
        "items": items if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.post("/tasks/create")
def tasks_create(body: TaskCreateRequest):
    settings = get_settings()
    ok, task, message = create_task_record(settings.database_url, body.model_dump())

    if not ok and message == "external_task_id уже существует":
        return JSONResponse(
            status_code=409,
            content={
                "service": "AA",
                "operation": "create_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "service": "AA",
            "operation": "create_task",
            "status": "ok" if ok else "error",
            "message": message,
            "task": task if ok else None,
        },
    )

def _build_az_brief_for_inventory_locations_fix(task_id: int, payload: dict | None) -> dict:
    """
    Формирует brief для АЗ по боевой задаче:
    вкладка 'Остатки' -> корректный вывод колонок 'Склад' и 'Зона размещения'.
    """
    payload = payload or {}

    screen = payload.get("screen") or "Остатки"
    title = payload.get("title") or "Вывод склада и зоны размещения на вкладке Остатки"
    user_request = str(payload.get("user_request") or "").strip()

    target_columns = payload.get("target_columns") or ["Склад", "Зона размещения"]
    acceptance_criteria = payload.get("acceptance_criteria") or []
    notes_for_az = payload.get("notes_for_az") or []

    # Нормализуем в списки строк (на случай если в payload пришло что-то кривое)
    if not isinstance(target_columns, list):
        target_columns = [str(target_columns)]
    target_columns = [str(x) for x in target_columns]

    if not isinstance(acceptance_criteria, list):
        acceptance_criteria = [str(acceptance_criteria)]
    acceptance_criteria = [str(x) for x in acceptance_criteria]

    if not isinstance(notes_for_az, list):
        notes_for_az = [str(notes_for_az)]
    notes_for_az = [str(x) for x in notes_for_az]

    az_brief = {
        "brief_version": "v1",
        "task_id": task_id,
        "task_type": "ozonator_inventory_locations_fix",
        "screen": screen,
        "title": title,
        "problem_summary": user_request,
        "target_columns": target_columns,
        "acceptance_criteria": acceptance_criteria,
        "notes_for_az": notes_for_az,
        "expected_output_from_az": {
            "must_define": [
                "Источник данных для поля 'Зона размещения' в текущем коде",
                "Источник данных для поля 'Склад' в текущем коде",
                "Место в коде, где формируются строки вкладки 'Остатки'",
                "Логика группировки строк, если у товара несколько зон размещения"
            ],
            "must_return": [
                "Короткий технический план исправления",
                "Список файлов/модулей, которые нужно менять",
                "Риски и что проверить после правки"
            ]
        }
    }

    return {
        "routed_to": "AZ",
        "mode": "az_brief_v1",
        "task_type": "ozonator_inventory_locations_fix",
        "task_id": task_id,
        "az_brief": az_brief,
        "next_action": "az_prepare_fix_plan"
    }
@app.post("/aa/run-task/{task_id}")
def aa_run_task(task_id: int):
    settings = get_settings()

    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_run_started",
        level="info",
        message="AA начал обработку задачи",
        meta={
            "target_agent": task["target_agent"],
            "task_type": task["task_type"],
        },
    )

    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    payload = task.get("payload") or {}

    if task.get("task_type") == "ozonator_inventory_locations_fix":
        execution_result = _build_az_brief_for_inventory_locations_fix(task_id, payload)
    else:
        execution_result = {
            "routed_to": "AZ",
            "mode": "aa_handoff_v1",
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "handoff_ready": True,
            "next_agent": "AZ",
            "aa_status": "routed_to_az",
            "note": "AA подготовил задачу и передал handoff в AZ",
        }

    if isinstance(execution_result, dict):
        execution_result["handoff_ready"] = True
        execution_result["next_agent"] = "AZ"
        execution_result["aa_executor"] = "AA"

    ok_set, task, message_set = set_task_result(
        settings.database_url,
        task_id=task_id,
        result=execution_result,
        error_message=None,
    )
    if not ok_set:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message_set,
                "task": None,
            },
        )

    ok, task, message = update_task_status(settings.database_url, task_id, "AA_ROUTED")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_routed_to_az",
        level="info",
        message="AA подготовил handoff и маршрутизировал задачу в AZ",
        meta=execution_result,
    )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AA",
            "operation": "aa_run_task",
            "status": "ok",
            "message": "AA подготовил handoff в AZ",
            "task": task,
            "execution_result": execution_result,
        },
    )

@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)

    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AA",
                "operation": "get_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AA",
            "operation": "get_task",
            "status": "ok",
            "message": "OK",
            "task": task,
        },
    )


@app.get("/tasks/{task_id}/logs")
def get_task_logs_endpoint(task_id: int):
    settings = get_settings()

    # сначала убедимся, что задача существует
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AA",
                "operation": "get_task_logs",
                "status": "error",
                "message": message,
                "task_id": task_id,
                "logs": None,
            },
        )

    ok, logs, message = get_task_logs(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "get_task_logs",
                "status": "error",
                "message": message,
                "task_id": task_id,
                "logs": None,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AA",
            "operation": "get_task_logs",
            "status": "ok",
            "message": "OK",
            "task_id": task_id,
            "count": len(logs),
            "logs": logs,
        },
    )
