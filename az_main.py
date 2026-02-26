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


def _build_az_brief(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    current_result = task.get("result") or {}
    current_result = current_result if isinstance(current_result, dict) else {}

    title = str(payload.get("title") or "Рабочая задача")
    screen = str(payload.get("screen") or "Не указано")
    task_type = str(task.get("task_type") or "unknown")
    user_request = str(payload.get("user_request") or "").strip()

    acceptance_criteria = payload.get("acceptance_criteria") or []
    if not isinstance(acceptance_criteria, list):
        acceptance_criteria = [str(acceptance_criteria)]
    acceptance_criteria = [str(x) for x in acceptance_criteria]

    notes = payload.get("notes") or []
    if not isinstance(notes, list):
        notes = [str(notes)]
    notes = [str(x) for x in notes]

    restrictions = payload.get("restrictions") or payload.get("do_not_touch") or []
    if not isinstance(restrictions, list):
        restrictions = [str(restrictions)]
    restrictions = [str(x) for x in restrictions]

    known_inputs = []
    for key in ("title", "screen", "user_request", "acceptance_criteria", "notes"):
        if payload.get(key):
            known_inputs.append(key)

    technical_plan = [
        "Проверить входные данные задачи и зафиксировать цель на языке результата.",
        "Определить область изменений (файл/модуль/экран/endpoint), не выходя за ограничения.",
        "Подготовить постановку для AS: что именно нужно сделать и как проверить, что готово.",
    ]

    qa_checklist = [
        "Проверить, что результат соответствует цели и acceptance criteria.",
        "Проверить, что ограничения не нарушены.",
        "Проверить, что артефакты AS можно применить/использовать без дополнительных догадок.",
    ]

    missing_inputs = []
    if not user_request:
        missing_inputs.append("Не заполнено payload.user_request")

    if not acceptance_criteria:
        acceptance_criteria = [
            "Результат соответствует задаче пользователя.",
            "Нет явных конфликтов с указанными ограничениями.",
        ]

    return {
        "brief_version": "az_brief_v1",
        "task_id": task_id,
        "task_type": task_type,
        "title": title,
        "screen": screen,
        "goal": user_request or title,
        "scope": screen,
        "done_definition": acceptance_criteria,
        "restrictions": restrictions,
        "known_inputs": known_inputs,
        "missing_inputs": missing_inputs,
        "questions_for_user": [],
        "technical_plan": technical_plan,
        "post_fix_checks": qa_checklist,
        "notes": notes,
        "previous_state": {
            "prev_status": task.get("status"),
            "has_result": bool(current_result),
        },
        "generated_at": _now_iso(),
    }


def _az_handoff_allowed(task: dict[str, Any]) -> tuple[bool, str]:
    target_agent = (task.get("target_agent") or "").upper()
    task_status = (task.get("status") or "").upper()

    if target_agent == "AZ":
        return True, ""

    if task_status in {"NEW", "IN_PROGRESS"}:
        return True, ""

    return (
        False,
        "AZ не может принять задачу: ожидается target_agent='AZ' или статус NEW/IN_PROGRESS.",
    )


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

    allowed, deny_message = _az_handoff_allowed(task)
    if not allowed:
        return JSONResponse(
            status_code=400,
            content={
                "service": "AZ",
                "operation": "az_run_task",
                "status": "error",
                "message": deny_message,
                "task": task,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AZ",
        event_type="task_run_started",
        level="info",
        message="AZ начал подготовку brief",
        meta={
            "source_agent": task.get("source_agent"),
            "task_type": task.get("task_type"),
            "prev_status": task.get("status"),
        },
    )

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
        az_brief = _build_az_brief(task_id, task)
        prev_result = task.get("result") if isinstance(task.get("result"), dict) else {}

        merged_result = {
            **prev_result,
            "az_executor": "AZ",
            "az_fix_plan": az_brief,
            "az_status": "brief_ready",
            "handoff_ready": True,
            "next_agent": "AS",
            "az_completed_at": _now_iso(),
            "next_action": "az_brief_ready",
        }

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AZ",
            event_type="task_brief_prepared",
            level="info",
            message="AZ подготовил brief для AS",
            meta={
                "mode": "az_brief_v1",
                "task_type": task.get("task_type"),
                "next_agent": "AS",
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
            event_type="az_brief_ready",
            level="info",
            message="AZ завершил подготовку brief (BRIEF_READY)",
            meta={
                "mode": "az_brief_v1",
                "task_id": task_id,
                "next_action": "az_brief_ready",
                "from_status": "in_progress",
                "to_status": "BRIEF_READY",
                "az_status": "brief_ready",
                "next_agent": "AS",
                "handoff_ready": True,
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
                    "mode": "az_brief_v1",
                    "task_id": task_id,
                    "next_action": "az_brief_ready",
                    "handoff_ready": True,
                    "next_agent": "AS",
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
