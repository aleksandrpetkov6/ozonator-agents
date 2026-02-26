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

app = FastAPI(title="Ozonator Agents AS")


# =========================
# Helpers
# =========================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_as_artifacts(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    current_result = task.get("result") or {}
    current_result = current_result if isinstance(current_result, dict) else {}

    az_fix_plan = current_result.get("az_fix_plan") if isinstance(current_result.get("az_fix_plan"), dict) else {}
    title = (
        payload.get("title")
        or az_fix_plan.get("title")
        or "Рабочий артефакт"
    )
    screen = payload.get("screen") or az_fix_plan.get("screen") or "Не указано"
    target_columns = payload.get("target_columns") or az_fix_plan.get("target_columns") or []
    task_type = task.get("task_type") or "unknown"

    # MVP-артефакт: AS не правит внешний репозиторий, а собирает пакет для следующего этапа (AK/AA)
    # с понятной структурой и трассировкой на brief от AZ.
    implementation_steps = []
    for idx, step in enumerate(az_fix_plan.get("technical_plan") or [], start=1):
        implementation_steps.append({
            "step_no": idx,
            "title_ru": f"Шаг {idx}",
            "description_ru": str(step),
        })

    if not implementation_steps:
        implementation_steps = [
            {
                "step_no": 1,
                "title_ru": "Подготовка",
                "description_ru": "AS не получил технический план от AZ и собрал универсальную заготовку артефакта.",
            }
        ]

    artifact_text_lines = [
        f"Задача #{task_id}",
        f"Тип: {task_type}",
        f"Экран: {screen}",
        f"Название: {title}",
        "",
        "Колонки-цели:",
    ]
    if target_columns:
        artifact_text_lines.extend([f"- {c}" for c in target_columns])
    else:
        artifact_text_lines.append("- (не указаны)")

    artifact_text_lines.extend([
        "",
        "План реализации (из brief AZ):",
    ])
    for item in implementation_steps:
        artifact_text_lines.append(f"{item['step_no']}. {item['description_ru']}")

    checks = az_fix_plan.get("post_fix_checks") or payload.get("acceptance_criteria") or []
    artifact_text_lines.extend([
        "",
        "Проверки после исправления:",
    ])
    if checks:
        artifact_text_lines.extend([f"- {c}" for c in checks])
    else:
        artifact_text_lines.append("- Проверки не заданы")

    return {
        "artifacts_version": "as_artifacts_v1",
        "task_id": task_id,
        "task_type": task_type,
        "generated_at": _now_iso(),
        "source_brief": {
            "az_status": current_result.get("az_status"),
            "brief_version": az_fix_plan.get("brief_version"),
            "handoff_ready": current_result.get("handoff_ready"),
            "next_agent_before_as": current_result.get("next_agent"),
        },
        "artifact_bundle": [
            {
                "artifact_name": "implementation_plan.txt",
                "artifact_kind": "text/mock",
                "produced_by": "AS",
                "description_ru": "Пакет AS для AK: план реализации и чек-лист по brief от AZ.",
                "content_text": "\n".join(artifact_text_lines),
            },
            {
                "artifact_name": "implementation_steps.json",
                "artifact_kind": "json/mock",
                "produced_by": "AS",
                "description_ru": "Структурированный список шагов реализации для дальнейшей проверки AK.",
                "content_json": implementation_steps,
            },
        ],
        "acceptance_checks": checks,
        "notes": [
            "AS собрал mock-артефакты внутри ozonator-agents (MVP-контур).",
            "Следующий этап — AK проверяет полноту, риски и соответствие brief/чек-листу.",
        ],
    }


def _as_handoff_allowed(task: dict[str, Any]) -> tuple[bool, str]:
    """
    В MVP допускаем запуск AS в двух случаях:
    1) target_agent == AS
    2) после AZ уже стоит BRIEF_READY и в result есть handoff_ready + next_agent=AS

    Это уменьшает ручные действия и не требует отдельного метода смены target_agent.
    """
    target_agent = (task.get("target_agent") or "").upper()
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    next_agent = (result.get("next_agent") or "").upper()
    handoff_ready = bool(result.get("handoff_ready"))
    task_status = (task.get("status") or "").upper()

    if target_agent == "AS":
        return True, ""

    if task_status == "BRIEF_READY" and handoff_ready and next_agent == "AS":
        return True, ""

    return (
        False,
        (
            "AS не может принять задачу: ожидается target_agent='AS' "
            "или handoff от AZ (status=BRIEF_READY, handoff_ready=true, next_agent='AS')."
        ),
    )


# =========================
# Base / Health
# =========================

@app.get("/")
def root():
    return {
        "service": "AS",
        "status": "ok",
        "message": "Ozonator Agents AS service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "AS"}


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, detail = check_postgres(settings.database_url)
    return {
        "service": "AS",
        "component": "db",
        "ok": ok,
        "detail": detail,
    }


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, detail = check_redis(settings.redis_url)
    return {
        "service": "AS",
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
        "service": "AS",
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
                "service": "AS",
                "operation": "get_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AS",
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
                "service": "AS",
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
                "service": "AS",
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
            "service": "AS",
            "operation": "get_task_logs",
            "status": "ok",
            "message": "OK",
            "task_id": task_id,
            "count": len(logs),
            "logs": logs,
        },
    )


# =========================
# AS runner
# =========================

@app.post("/as/run-task/{task_id}")
def as_run_task(task_id: int):
    settings = get_settings()

    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AS",
                "operation": "as_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    allowed, deny_message = _as_handoff_allowed(task)
    if not allowed:
        return JSONResponse(
            status_code=400,
            content={
                "service": "AS",
                "operation": "as_run_task",
                "status": "error",
                "message": deny_message,
                "task": task,
            },
        )

    # Лог: старт обработки
    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AS",
        event_type="task_run_started",
        level="info",
        message="AS начал сборку артефактов",
        meta={
            "source_agent": task.get("source_agent"),
            "task_type": task.get("task_type"),
            "prev_status": task.get("status"),
        },
    )

    # Статус -> in_progress
    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AS",
                "operation": "as_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    try:
        as_artifacts = _build_as_artifacts(task_id, task)

        prev_result = task.get("result") if isinstance(task.get("result"), dict) else {}
        merged_result = {
            **prev_result,
            "as_executor": "AS",
            "as_artifacts": as_artifacts,
            "as_status": "artifacts_ready",
            "handoff_ready": True,
            "next_agent": "AK",
            "as_completed_at": _now_iso(),
            "next_action": "as_artifacts_ready",
        }

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AS",
            event_type="task_artifacts_prepared",
            level="info",
            message="AS собрал артефакты по brief от AZ",
            meta={
                "mode": "as_artifacts_v1",
                "task_type": task.get("task_type"),
                "artifact_count": len(as_artifacts.get("artifact_bundle") or []),
                "next_agent": "AK",
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
                    "service": "AS",
                    "operation": "as_run_task",
                    "status": "error",
                    "message": message_set,
                    "task": None,
                },
            )

        ok, task, message = update_task_status(settings.database_url, task_id, "ARTIFACTS_READY")
        if not ok:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "AS",
                    "operation": "as_run_task",
                    "status": "error",
                    "message": message,
                    "task": None,
                },
            )

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AS",
            event_type="as_artifacts_ready",
            level="info",
            message="AS завершил сборку артефактов (ARTIFACTS_READY)",
            meta={
                "mode": "as_artifacts_v1",
                "task_id": task_id,
                "next_action": "as_artifacts_ready",
                "from_status": "in_progress",
                "to_status": "ARTIFACTS_READY",
                "as_status": "artifacts_ready",
                "next_agent": "AK",
                "handoff_ready": True,
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "service": "AS",
                "operation": "as_run_task",
                "status": "ok",
                "message": "Задача обработана",
                "task": task,
                "execution_result": {
                    "mode": "as_artifacts_v1",
                    "task_id": task_id,
                    "next_action": "as_artifacts_ready",
                    "handoff_ready": True,
                    "next_agent": "AK",
                },
            },
        )

    except Exception as e:
        err = f"AS error: {e}"

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
            actor_agent="AS",
            event_type="task_run_failed",
            level="error",
            message="AS завершил обработку с ошибкой",
            meta={"error": err},
        )

        return JSONResponse(
            status_code=500,
            content={
                "service": "AS",
                "operation": "as_run_task",
                "status": "error",
                "message": err,
                "task": None,
            },
        )
