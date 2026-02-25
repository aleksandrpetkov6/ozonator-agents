from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from db.health import check_postgres, check_redis
from db.tasks import (
    get_task_logs,
    get_task_record,
    set_task_result,
    update_task_status,
    write_orchestration_log,
)

app = FastAPI(title="Ozonator Agents AZ")


# =========================
# Helpers
# =========================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_az_fix_plan(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    current_result = task.get("result") or {}
    aa_brief = current_result.get("az_brief") if isinstance(current_result, dict) else None
    aa_brief = aa_brief if isinstance(aa_brief, dict) else {}

    title = (
        payload.get("title")
        or aa_brief.get("title")
        or "План исправления"
    )
    screen = payload.get("screen") or aa_brief.get("screen") or "Остатки"
    target_columns = payload.get("target_columns") or aa_brief.get("target_columns") or []
    acceptance_criteria = (
        payload.get("acceptance_criteria")
        or aa_brief.get("acceptance_criteria")
        or []
    )
    problem_summary = (
        payload.get("user_request")
        or aa_brief.get("problem_summary")
        or ""
    )

    # Если задача именно про вывод склада/зоны размещения — формируем предметный план.
    if task.get("task_type") == "ozonator_inventory_locations_fix":
        return {
            "brief_version": "az_fix_plan_v1",
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "screen": screen,
            "title": title,
            "generated_at": _now_iso(),
            "target_columns": target_columns,
            "problem_summary": problem_summary,
            "must_define": [
                "Источник данных для поля 'Зона размещения' в текущем коде Озонатора",
                "Источник данных для поля 'Склад' в текущем коде Озонатора",
                "Место в коде, где формируются строки вкладки 'Остатки'",
                "Логика группировки строк, если у товара несколько зон размещения",
            ],
            "technical_plan": [
                "Найти backend-источник данных для размещения товара (склад/зона) и проверить, в каком виде данные приходят в локальное хранилище.",
                "Найти mapper/DTO/преобразователь строк вкладки 'Остатки' и убрать подстановку заглушек ('—', пусто, 'Нет данных синхронизации'), если реальные данные присутствуют.",
                "Добавить развертывание записей по зонам размещения: если у товара несколько зон, выводить несколько строк с одинаковым товаром, но разными значениями зоны.",
                "Для каждой строки с зоной указывать любой корректный склад, соответствующий этой зоне (из той же записи источника).",
                "Добавить стабильную сортировку/группировку, чтобы строки одного товара с разными зонами шли подряд.",
                "Проверить рендер таблицы: убедиться, что UI не перетирает реальные значения fallback-текстом.",
            ],
            "files_modules_to_change": [
                "Модуль синхронизации/получения остатков и размещения (backend/API layer)",
                "Модуль локального хранения/чтения данных по остаткам и размещению (DB/repository layer)",
                "Преобразователь данных в строки вкладки 'Остатки' (mapper/DTO)",
                "UI-таблица вкладки 'Остатки' (рендер колонок 'Склад' и 'Зона размещения')",
                "Логика сортировки/группировки строк в таблице 'Остатки'",
            ],
            "risks": [
                "Данные по складу/зоне могут храниться в другой структуре, чем текущие поля таблицы — потребуется адаптер/маппинг.",
                "При развертывании нескольких зон размещения может сломаться текущая группировка/виртуализация списка, если строки схлопываются по SKU.",
                "Возможны частично пустые данные (есть зона, но нет склада) — нужен аккуратный fallback только для реально отсутствующих значений.",
            ],
            "post_fix_checks": acceptance_criteria or [
                "В столбце 'Зона размещения' у каждого товара отображается реальное значение",
                "В столбце 'Склад' отображается корректное значение, соответствующее зоне",
                "Если у товара несколько зон размещения, строки идут рядом",
            ],
            "note": "AZ сформировал технический план по brief от AA. Для точных путей файлов нужен код репозитория Озонатора (не ozonator-agents).",
        }

    # Универсальный fallback для других типов задач
    return {
        "brief_version": "az_fix_plan_v1",
        "task_id": task_id,
        "task_type": task.get("task_type"),
        "generated_at": _now_iso(),
        "title": title,
        "screen": screen,
        "problem_summary": problem_summary,
        "technical_plan": [
            "Прочитать payload и brief от AA",
            "Определить источник данных и точки формирования UI/ответа",
            "Подготовить план правки по слоям: source -> mapping -> UI -> checks",
        ],
        "files_modules_to_change": [
            "backend source", "data mapping", "UI renderer"
        ],
        "risks": ["Нужна детализация по коду целевого репозитория"],
        "post_fix_checks": acceptance_criteria,
    }


# =========================
# Base / Health
# =========================

@app.get("/")
def root():
    return {
        "service": "AZ",
        "status": "ok",
        "message": "Ozonator Agents AZ service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "AZ"}


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, detail = check_postgres(settings.database_url)
    return {
        "service": "AZ",
        "component": "db",
        "ok": ok,
        "detail": detail,
    }


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, detail = check_redis(settings.redis_url)
    return {
        "service": "AZ",
        "component": "redis",
        "ok": ok,
        "detail": detail,
    }


@app.get("/health/all")
def health_all():
    settings = get_settings()
    ok_db, db_detail = check_postgres(settings.database_url)
    ok_redis, redis_detail = check_redis(settings.redis_url)

    return {
        "service": "AZ",
        "status": "ok" if (ok_db and ok_redis) else "degraded",
        "components": {
            "db": {"ok": ok_db, "detail": db_detail},
            "redis": {"ok": ok_redis, "detail": redis_detail},
        },
    }


# =========================
# Tasks read endpoints
# =========================

@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)

    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AZ",
                "operation": "get_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AZ",
            "operation": "get_task",
            "status": "ok",
            "message": "OK",
            "task": task,
        },
    )


@app.get("/tasks/{task_id}/logs")
def get_task_logs_endpoint(task_id: int):
    settings = get_settings()

    ok, _task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AZ",
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
                "service": "AZ",
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
            "service": "AZ",
            "operation": "get_task_logs",
            "status": "ok",
            "message": "OK",
            "task_id": task_id,
            "count": len(logs),
            "logs": logs,
        },
    )


# =========================
# AZ runner
# =========================

@app.post("/az/run-task/{task_id}")
def az_run_task(task_id: int):
    settings = get_settings()

    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    target_agent = (task.get("target_agent") or "").upper()
    if target_agent != "AZ":
        return JSONResponse(
            status_code=400,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "error",
                "message": f"Задача предназначена для {target_agent or 'UNKNOWN'}, а не для AZ",
                "task": task,
            },
        )

    # Лог: старт обработки
    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AZ",
        event_type="task_run_started",
        level="info",
        message="AZ начал обработку задачи",
        meta={
            "source_agent": task.get("source_agent"),
            "task_type": task.get("task_type"),
        },
    )

    # Статус -> in_progress
    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    try:
        az_fix_plan = _build_az_fix_plan(task_id, task)

        prev_result = task.get("result") if isinstance(task.get("result"), dict) else {}
        merged_result = {
            **prev_result,
            "az_executor": "AZ",
            "az_fix_plan": az_fix_plan,
            "az_status": "brief_ready",
            "az_completed_at": _now_iso(),
            "next_action": "az_fix_plan_ready",
            "handoff_ready": True,
            "next_agent": "AS",
        }

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AZ",
            event_type="task_plan_prepared",
            level="info",
            message="AZ сформировал технический план исправления",
            meta={
                "mode": "az_fix_plan_v1",
                "task_type": task.get("task_type"),
                "screen": az_fix_plan.get("screen"),
                "target_columns": az_fix_plan.get("target_columns", []),
            },
        )

        ok_set, task, message_set = set_task_result(
            settings.database_url,
            task_id=task_id,
            result=merged_result,
            error_message=None,
        )
        if not ok_set:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "AZ",
                    "operation": "az_run_task",
                    "status": "error",
                    "message": message_set,
                    "task": None,
                },
            )

        ok, task, message = update_task_status(settings.database_url, task_id, "BRIEF_READY")
        if not ok:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "AZ",
                    "operation": "az_run_task",
                    "status": "error",
                    "message": message,
                    "task": None,
                },
            )

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AZ",
            event_type="task_run_finished",
            level="info",
            message="AZ завершил обработку задачи",
            meta={
                "mode": "az_fix_plan_v1",
                "task_id": task_id,
                "next_action": "az_fix_plan_ready",
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "ok",
                "message": "Задача обработана",
                "task": task,
                "execution_result": {
                    "mode": "az_fix_plan_v1",
                    "task_id": task_id,
                    "next_action": "az_fix_plan_ready",
                },
            },
        )

    except Exception as e:
        err = f"AZ error: {e}"

        set_task_result(
            settings.database_url,
            task_id=task_id,
            result=(task.get("result") if isinstance(task.get("result"), dict) else task.get("result")),
            error_message=err,
        )
        update_task_status(settings.database_url, task_id, "failed")

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AZ",
            event_type="task_run_failed",
            level="error",
            message="AZ завершил обработку с ошибкой",
            meta={"error": err},
        )

        return JSONResponse(
            status_code=500,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "error",
                "message": err,
                "task": None,
            },
        )
