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

app = FastAPI(title="Ozonator Agents AK")


# =========================
# Helpers
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_ak_review(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    as_artifacts = result.get("as_artifacts") if isinstance(result.get("as_artifacts"), dict) else {}
    az_fix_plan = result.get("az_fix_plan") if isinstance(result.get("az_fix_plan"), dict) else {}

    available_artifact_keys = []
    for key in [
        "as_artifacts",
        "as_patch_text",
        "as_files",
        "as_delivery",
        "as_fix_package",
        "as_output",
    ]:
        if key in result:
            available_artifact_keys.append(key)

    checks = [
        {
            "check": "brief_from_az_present",
            "ok": "az_fix_plan" in result,
            "details": "Найден az_fix_plan от AZ" if "az_fix_plan" in result else "az_fix_plan отсутствует",
        },
        {
            "check": "artifacts_from_as_present",
            "ok": bool(available_artifact_keys),
            "details": (
                f"Найдены артефакты AS: {', '.join(available_artifact_keys)}"
                if available_artifact_keys
                else "Артефакты AS не найдены по ожидаемым ключам"
            ),
        },
        {
            "check": "as_handoff_ready_flag",
            "ok": bool(result.get("handoff_ready")),
            "details": "AS пометил handoff_ready=true" if result.get("handoff_ready") else "AS не пометил handoff_ready=true",
        },
        {
            "check": "as_next_agent_is_ak",
            "ok": (result.get("next_agent") == "AK"),
            "details": (
                "AS передал next_agent=AK"
                if result.get("next_agent") == "AK"
                else f"Ожидался next_agent=AK, получено: {result.get('next_agent')!r}"
            ),
        },
    ]

    all_ok = all(bool(item.get("ok")) for item in checks)

    target_columns = az_fix_plan.get("target_columns", [])
    artifacts_preview = {}
    if as_artifacts:
        for k, v in as_artifacts.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                artifacts_preview[k] = v

    return {
        "review_version": "ak_review_v1",
        "task_id": task_id,
        "reviewed_at": _now_iso(),
        "task_type": task.get("task_type"),
        "screen": (task.get("payload") or {}).get("screen"),
        "target_columns": target_columns,
        "decision": "approved" if all_ok else "needs_attention",
        "summary": (
            "AK проверил handoff от AS. Артефакты готовы к использованию."
            if all_ok
            else "AK проверил handoff от AS. Есть замечания, требуется доработка/уточнение."
        ),
        "checks": checks,
        "artifacts_preview": artifacts_preview,
        "available_artifact_keys": available_artifact_keys,
        "notes": [
            "Это MVP-проверка контура AK.",
            "На следующем шаге можно добавить более строгую валидацию содержимого артефактов (patch/file checks).",
        ],
    }


# =========================
# Base / Health
# =========================
@app.get("/")
def root():
    return {
        "service": "AK",
        "status": "ok",
        "message": "Ozonator Agents AK service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "AK"}


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, detail = check_postgres(settings.database_url)
    return {
        "service": "AK",
        "component": "db",
        "ok": ok,
        "detail": detail,
    }


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, detail = check_redis(settings.redis_url)
    return {
        "service": "AK",
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
        "service": "AK",
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
                "service": "AK",
                "operation": "get_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "service": "AK",
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
                "service": "AK",
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
                "service": "AK",
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
            "service": "AK",
            "operation": "get_task_logs",
            "status": "ok",
            "message": "OK",
            "task_id": task_id,
            "count": len(logs),
            "logs": logs,
        },
    )


# =========================
# AK runner
# =========================
@app.post("/ak/run-task/{task_id}")
def ak_run_task(task_id: int):
    settings = get_settings()

    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    target_agent = (task.get("target_agent") or "").upper()

    # Allow AK to accept handoff even if the original target_agent is different (e.g., AZ).
    # Handoff contract: status=ARTIFACTS_READY + result.handoff_ready=true + result.next_agent='AK'
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    next_agent = (result.get("next_agent") or "").upper()
    handoff_ready = bool(result.get("handoff_ready"))
    task_status = (task.get("status") or "").upper()

    allowed_handoff = (task_status == "ARTIFACTS_READY" and handoff_ready and next_agent == "AK")

    if target_agent and target_agent != "AK" and not allowed_handoff:
        return JSONResponse(
            status_code=400,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "error",
                "message": (
                    f"AK не может принять задачу: target_agent={target_agent}. "
                    "Ожидается target_agent='AK' или handoff от AS "
                    "(status=ARTIFACTS_READY, handoff_ready=true, next_agent='AK')."
                ),
                "task": task,
            },
        )

    current_status = task.get("status")
    if current_status != "ARTIFACTS_READY":
        return JSONResponse(
            status_code=400,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "error",
                "message": f"Для запуска AK нужен статус ARTIFACTS_READY, сейчас: {current_status}",
                "task": task,
            },
        )

    # Лог: старт обработки
    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AK",
        event_type="task_run_started",
        level="info",
        message="AK начал проверку артефактов",
        meta={
            "source_agent": task.get("source_agent"),
            "task_type": task.get("task_type"),
            "from_status": current_status,
        },
    )

    # Статус -> in_progress (короткий рабочий статус на время проверки)
    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    try:
        ak_review = _build_ak_review(task_id, task)
        prev_result = task.get("result") if isinstance(task.get("result"), dict) else {}

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AK",
            event_type="ak_review_started",
            level="info",
            message="AK начал проверку handoff от AS",
            meta={
                "mode": "ak_review_v1",
                "task_type": task.get("task_type"),
                "screen": (task.get("payload") or {}).get("screen"),
            },
        )

        final_decision = ak_review.get("decision")
        final_task_status = "DONE" if final_decision == "approved" else "REVIEW_NEEDS_ATTENTION"
        final_ak_status = "review_completed" if final_decision == "approved" else "review_needs_attention"

        merged_result = {
            **prev_result,
            "ak_executor": "AK",
            "ak_review": ak_review,
            "ak_status": final_ak_status,
            "ak_completed_at": _now_iso(),
            "handoff_ready": False,
            "next_agent": None,
            "next_action": "task_done" if final_decision == "approved" else "return_to_as",
        }

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
                    "service": "AK",
                    "operation": "ak_run_task",
                    "status": "error",
                    "message": message_set,
                    "task": None,
                },
            )

        ok, task, message = update_task_status(settings.database_url, task_id, final_task_status)
        if not ok:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "AK",
                    "operation": "ak_run_task",
                    "status": "error",
                    "message": message,
                    "task": None,
                },
            )

        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AK",
            event_type="ak_review_completed",
            level="info" if final_decision == "approved" else "warning",
            message=(
                "AK завершил проверку: артефакты приняты"
                if final_decision == "approved"
                else "AK завершил проверку: есть замечания"
            ),
            meta={
                "mode": "ak_review_v1",
                "task_id": task_id,
                "decision": final_decision,
                "from_status": "in_progress",
                "to_status": final_task_status,
                "ak_status": final_ak_status,
                "next_agent": None,
                "handoff_ready": False,
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "ok",
                "message": "Проверка AK выполнена",
                "task": task,
                "execution_result": {
                    "mode": "ak_review_v1",
                    "task_id": task_id,
                    "decision": final_decision,
                    "ak_status": final_ak_status,
                    "task_status": final_task_status,
                    "handoff_ready": False,
                    "next_agent": None,
                },
            },
        )

    except Exception as e:
        err = f"AK error: {e}"

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
            actor_agent="AK",
            event_type="task_run_failed",
            level="error",
            message="AK завершил обработку с ошибкой",
            meta={"error": err},
        )

        return JSONResponse(
            status_code=500,
            content={
                "service": "AK",
                "operation": "ak_run_task",
                "status": "error",
                "message": err,
                "task": None,
            },
        )
