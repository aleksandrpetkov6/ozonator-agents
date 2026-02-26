import json
import os
from typing import Annotated, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

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
from db.tasks import (
    create_task_record,
    get_task_logs,
    get_task_record,
    set_task_result,
    update_task_status,
    write_orchestration_log,
)
from schemas.tasks import TaskCreateRequest

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



def _build_az_brief_for_inventory_locations_fix(task_id: int, payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    screen = payload.get("screen") or "Остатки"
    title = payload.get("title") or "Вывод склада и зоны размещения на вкладке Остатки"
    user_request = str(payload.get("user_request") or "").strip()
    target_columns = payload.get("target_columns") or ["Склад", "Зона размещения"]
    acceptance_criteria = payload.get("acceptance_criteria") or []
    notes_for_az = payload.get("notes_for_az") or []

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
                "Логика группировки строк, если у товара несколько зон размещения",
            ],
            "must_return": [
                "Короткий технический план исправления",
                "Список файлов/модулей, которые нужно менять",
                "Риски и что проверить после правки",
            ],
        },
    }

    return {
        "routed_to": "AZ",
        "mode": "az_brief_v1",
        "task_type": "ozonator_inventory_locations_fix",
        "task_id": task_id,
        "az_brief": az_brief,
        "next_action": "az_prepare_fix_plan",
    }



def _json_response(status_code: int, content: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=content)



def _read_json_response(raw_bytes: bytes) -> Any:
    if not raw_bytes:
        return None
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return None



def _post_json(url: str, payload: dict[str, Any] | None = None) -> tuple[bool, int, Any, str | None]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=120) as response:
            raw = response.read()
            return True, response.getcode(), _read_json_response(raw), None
    except urllib_error.HTTPError as e:
        raw = e.read()
        return False, e.code, _read_json_response(raw), f"HTTPError {e.code}"
    except urllib_error.URLError as e:
        return False, 0, None, f"URLError: {e.reason}"
    except Exception as e:
        return False, 0, None, f"{e.__class__.__name__}: {e}"



def _agent_base_url(settings, agent_code: str) -> str:
    agent_code = agent_code.upper()
    if agent_code == "AZ":
        return settings.az_run_task_base_url
    if agent_code == "AS":
        return settings.as_run_task_base_url
    if agent_code == "AK":
        return settings.ak_run_task_base_url
    raise ValueError(f"Неизвестный агент: {agent_code}")



def _agent_route(agent_code: str, task_id: int) -> str:
    agent_code = agent_code.lower()
    return f"/{agent_code}/run-task/{task_id}"



def _call_agent(settings, task_id: int, agent_code: str) -> tuple[bool, dict[str, Any]]:
    base_url = _agent_base_url(settings, agent_code)
    url = f"{base_url}{_agent_route(agent_code, task_id)}"
    ok, status_code, body, error_message = _post_json(url, {})
    return (
        ok,
        {
            "agent": agent_code.upper(),
            "url": url,
            "http_status": status_code,
            "response": body if isinstance(body, dict) else None,
            "error": error_message,
        },
    )



def _normalize_execution_response(data: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(data, dict):
        return None, None

    task_data = data.get("task") if isinstance(data.get("task"), dict) else {}
    execution_result = data.get("execution_result") if isinstance(data.get("execution_result"), dict) else {}
    task_result = task_data.get("result") if isinstance(task_data.get("result"), dict) else {}

    next_agent = (
        execution_result.get("next_agent")
        or task_result.get("next_agent")
        or data.get("next_agent")
    )
    task_status = (
        execution_result.get("task_status")
        or task_data.get("status")
        or data.get("task_status")
    )

    return (
        str(next_agent).upper() if next_agent else None,
        str(task_status).upper() if task_status else None,
    )



def _merge_result_for_rework(settings, task_id: int) -> tuple[bool, str]:
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return False, message

    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    current_result = current_result if isinstance(current_result, dict) else {}
    rework_result = {
        **current_result,
        "handoff_ready": True,
        "next_agent": "AS",
        "next_action": "ak_return_to_as",
    }

    ok_set, _, message_set = set_task_result(
        settings.database_url,
        task_id=task_id,
        result=rework_result,
        error_message=None,
    )
    return ok_set, message_set



def _run_single_cycle(settings, task_id: int, cycle_no: int) -> dict[str, Any]:
    step_results: list[dict[str, Any]] = []

    ok_az, az_payload = _call_agent(settings, task_id, "AZ")
    step_results.append(az_payload)
    if not ok_az:
        return {
            "ok": False,
            "failed_at": "AZ",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AZ",
        }

    az_response = az_payload.get("response") if isinstance(az_payload.get("response"), dict) else {}
    next_agent, _task_status = _normalize_execution_response(az_response)
    if (az_response.get("status") != "ok") or next_agent not in {"AS", None}:
        return {
            "ok": False,
            "failed_at": "AZ",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AZ не сформировал корректный handoff в AS",
        }

    ok_as, as_payload = _call_agent(settings, task_id, "AS")
    step_results.append(as_payload)
    if not ok_as:
        return {
            "ok": False,
            "failed_at": "AS",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AS",
        }

    as_response = as_payload.get("response") if isinstance(as_payload.get("response"), dict) else {}
    next_agent, _task_status = _normalize_execution_response(as_response)
    if (as_response.get("status") != "ok") or next_agent not in {"AK", None}:
        return {
            "ok": False,
            "failed_at": "AS",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AS не сформировал корректный handoff в AK",
        }

    ok_ak, ak_payload = _call_agent(settings, task_id, "AK")
    step_results.append(ak_payload)
    if not ok_ak:
        return {
            "ok": False,
            "failed_at": "AK",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AK",
        }

    ak_response = ak_payload.get("response") if isinstance(ak_payload.get("response"), dict) else {}
    _next_agent, task_status = _normalize_execution_response(ak_response)
    if ak_response.get("status") != "ok":
        return {
            "ok": False,
            "failed_at": "AK",
            "cycle_no": cycle_no,
            "steps": step_results,
            "message": "AK вернул ошибку при проверке",
        }

    return {
        "ok": True,
        "cycle_no": cycle_no,
        "steps": step_results,
        "final_status": task_status,
        "decision": (
            ((ak_response.get("execution_result") or {}).get("decision"))
            if isinstance(ak_response.get("execution_result"), dict)
            else None
        ),
    }



def _run_full_orchestration(settings, task_id: int) -> dict[str, Any]:
    cycles: list[dict[str, Any]] = []
    max_cycles = max(1, int(settings.aa_max_rework_cycles))

    for cycle_no in range(1, max_cycles + 1):
        cycle_result = _run_single_cycle(settings, task_id, cycle_no)
        cycles.append(cycle_result)

        if not cycle_result.get("ok"):
            return {
                "orchestration_status": "failed",
                "failed_at": cycle_result.get("failed_at"),
                "message": cycle_result.get("message"),
                "cycles": cycles,
            }

        final_status = (cycle_result.get("final_status") or "").upper()
        if final_status == "DONE":
            return {
                "orchestration_status": "done",
                "message": "AA автоматически провел задачу по цепочке AZ -> AS -> AK",
                "cycles": cycles,
            }

        if final_status == "REVIEW_NEEDS_ATTENTION":
            if cycle_no >= max_cycles:
                return {
                    "orchestration_status": "review_needs_attention",
                    "message": "AA выполнил автоматический цикл, но задача осталась на доработке",
                    "cycles": cycles,
                }

            ok_rework, rework_message = _merge_result_for_rework(settings, task_id)
            write_orchestration_log(
                settings.database_url,
                task_id=task_id,
                actor_agent="AA",
                event_type="aa_rework_requested",
                level="warning" if ok_rework else "error",
                message=(
                    "AA подготовил автоматический возврат в AS после замечаний AK"
                    if ok_rework
                    else "AA не смог подготовить автоматический возврат в AS"
                ),
                meta={
                    "cycle_no": cycle_no,
                    "rework_result": "ok" if ok_rework else "error",
                    "message": rework_message,
                },
            )
            if not ok_rework:
                return {
                    "orchestration_status": "failed",
                    "failed_at": "AA_REWORK",
                    "message": rework_message,
                    "cycles": cycles,
                }
            continue

        return {
            "orchestration_status": "unknown_final_status",
            "message": f"AA получил неожиданный финальный статус: {final_status or 'None'}",
            "cycles": cycles,
        }

    return {
        "orchestration_status": "review_needs_attention",
        "message": "AA исчерпал лимит автоматических циклов доработки",
        "cycles": cycles,
    }


@app.post("/aa/run-task/{task_id}")
def aa_run_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return _json_response(
            404 if message == "Задача не найдена" else 503,
            {
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
        return _json_response(
            503,
            {
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
        return _json_response(
            503,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message_set,
                "task": None,
            },
        )

    ok, task, message = update_task_status(settings.database_url, task_id, "AA_ROUTED")
    if not ok:
        return _json_response(
            503,
            {
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

    if not settings.aa_auto_orchestration_enabled:
        return _json_response(
            200,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "ok",
                "message": "AA подготовил handoff в AZ",
                "task": task,
                "execution_result": execution_result,
            },
        )

    orchestration_result = _run_full_orchestration(settings, task_id)
    level = "info" if orchestration_result.get("orchestration_status") == "done" else "warning"
    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="aa_auto_orchestration_completed",
        level=level,
        message="AA завершил автоматическую оркестрацию цепочки",
        meta={
            "orchestration_status": orchestration_result.get("orchestration_status"),
            "cycles": len(orchestration_result.get("cycles") or []),
        },
    )

    ok, final_task, final_message = get_task_record(settings.database_url, task_id)
    if not ok:
        return _json_response(
            503,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": final_message,
                "task": None,
                "execution_result": orchestration_result,
            },
        )

    return _json_response(
        200,
        {
            "service": "AA",
            "operation": "aa_run_task",
            "status": "ok",
            "message": "AA выполнил автоматический прогон цепочки",
            "task": final_task,
            "execution_result": {
                **execution_result,
                "orchestration": orchestration_result,
            },
        },
    )


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return _json_response(
            404 if message == "Задача не найдена" else 503,
            {
                "service": "AA",
                "operation": "get_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return _json_response(
        200,
        {
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
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return _json_response(
            404 if message == "Задача не найдена" else 503,
            {
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
        return _json_response(
            503,
            {
                "service": "AA",
                "operation": "get_task_logs",
                "status": "error",
                "message": message,
                "task_id": task_id,
                "logs": None,
            },
        )

    return _json_response(
        200,
        {
            "service": "AA",
            "operation": "get_task_logs",
            "status": "ok",
            "message": "OK",
            "task_id": task_id,
            "count": len(logs),
            "logs": logs,
        },
    )
