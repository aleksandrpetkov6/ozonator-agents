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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []

    items = value if isinstance(value, list) else [value]
    result: list[str] = []

    for item in items:
        text = str(item).strip()
        if text:
            result.append(text)

    return result


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        text = _normalize_text(item)
        if not text:
            continue

        marker = text.casefold()
        if marker in seen:
            continue

        seen.add(marker)
        result.append(text)

    return result


def _limit_items(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return items[:limit]


def _inline_list(items: list[str], limit: int = 3) -> str:
    prepared: list[str] = []

    for item in items:
        text = _normalize_text(item).rstrip(" .;:")
        if text:
            prepared.append(text)

    prepared = _limit_items(_dedupe_keep_order(prepared), limit)
    return "; ".join(prepared)


def _collect_named_items(*sources: Any, keys: list[str]) -> list[str]:
    collected: list[str] = []

    for source in sources:
        if not isinstance(source, dict):
            continue

        for key in keys:
            collected.extend(_normalize_str_list(source.get(key)))

    return _dedupe_keep_order(collected)


def _is_api_mapping_task(main_goal: str, task: dict[str, Any], az_fix_plan: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            main_goal,
            _normalize_text(task.get("task_type")),
            _normalize_text(az_fix_plan.get("scope")),
            _normalize_text(az_fix_plan.get("title")),
        ]
    ).casefold()

    api_markers = ["api", "endpoint", "эндпоинт"]
    mapping_markers = ["ключ", "связ", "payload", "response", "ozon"]

    return any(marker in haystack for marker in api_markers) and any(
        marker in haystack for marker in mapping_markers
    )


def _build_final_answer(task_id: int, task: dict[str, Any]) -> str:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    az_fix_plan = (
        current_result.get("az_fix_plan")
        if isinstance(current_result.get("az_fix_plan"), dict)
        else {}
    )

    user_request = _normalize_text(payload.get("user_request"))
    payload_brief = _normalize_text(payload.get("brief"))
    goal = _normalize_text(az_fix_plan.get("goal"))
    title = _normalize_text(payload.get("title")) or _normalize_text(az_fix_plan.get("title"))
    screen = _normalize_text(payload.get("screen")) or _normalize_text(az_fix_plan.get("screen"))
    target_columns = _dedupe_keep_order(
        _normalize_str_list(payload.get("target_columns"))
        or _normalize_str_list(az_fix_plan.get("target_columns"))
    )
    technical_plan = _dedupe_keep_order(_normalize_str_list(az_fix_plan.get("technical_plan")))
    checks = _dedupe_keep_order(
        _normalize_str_list(az_fix_plan.get("post_fix_checks"))
        or _normalize_str_list(payload.get("acceptance_criteria"))
    )
    restrictions = _dedupe_keep_order(
        _normalize_str_list(payload.get("restrictions"))
        or _normalize_str_list(az_fix_plan.get("restrictions"))
    )
    missing_inputs = _dedupe_keep_order(_normalize_str_list(az_fix_plan.get("missing_inputs")))
    known_inputs = _dedupe_keep_order(_normalize_str_list(az_fix_plan.get("known_inputs")))

    main_goal = user_request or payload_brief or goal or title or f"задача #{task_id}"

    endpoints = _collect_named_items(
        payload,
        az_fix_plan,
        keys=["endpoint", "endpoints", "api_endpoint", "api_endpoints", "target_endpoints"],
    )
    request_keys = _collect_named_items(
        payload,
        az_fix_plan,
        keys=["request_keys", "input_keys", "query_keys", "body_keys", "payload_keys"],
    )
    response_keys = _collect_named_items(
        payload,
        az_fix_plan,
        keys=["response_keys", "output_keys", "result_keys", "field_keys"],
    )
    link_keys = _collect_named_items(
        payload,
        az_fix_plan,
        keys=["link_keys", "binding_keys", "relation_keys", "entity_keys", "join_keys", "id_keys"],
    )

    if _is_api_mapping_task(main_goal, task, az_fix_plan):
        parts = [
            f"По задаче «{main_goal}» итоговый результат должен дать карту передачи данных от Ozon по API.",
        ]

        if endpoints:
            parts.append(f"Endpoint: {', '.join(_limit_items(endpoints, 6))}.")
        else:
            parts.append(
                "Нужно зафиксировать конкретные endpoint и методы вызова, через которые данные приходят от Ozon."
            )

        if request_keys:
            parts.append(f"Ключи запроса: {', '.join(_limit_items(request_keys, 8))}.")
        else:
            parts.append(
                "Нужно перечислить ключи запроса, которыми данные передаются в вызов."
            )

        if response_keys:
            parts.append(f"Ключи ответа: {', '.join(_limit_items(response_keys, 8))}.")
        else:
            parts.append(
                "Нужно перечислить ключи ответа, в которых Ozon возвращает нужные данные."
            )

        if link_keys:
            parts.append(f"Связка между вызовами: {', '.join(_limit_items(link_keys, 8))}.")
        else:
            parts.append(
                "Нужно указать связующие идентификаторы между вызовами и сущностями (например, ID товара, оффера, SKU, отправления или запроса)."
            )

        if screen and screen != "Не указано":
            parts.append(f"Контекст проверки: {screen}.")

        if technical_plan:
            parts.append(f"Рабочий фокус: {_inline_list(technical_plan, limit=2)}.")

        if restrictions:
            parts.append(f"Ограничения: {_inline_list(restrictions, limit=2)}.")

        if missing_inputs:
            parts.append(
                "Чтобы назвать конкретные endpoint и ключи без догадок, нужно дополнительно получить: "
                f"{', '.join(_limit_items(missing_inputs, 4))}."
            )
        elif known_inputs:
            parts.append(f"Опорные входные данные: {', '.join(_limit_items(known_inputs, 4))}.")

        return " ".join(parts)

    parts = [f"По задаче «{main_goal}» подготовлен воспроизводимый результат."]

    if title and title != main_goal:
        parts.append(f"Рабочее название: {title}.")

    if screen and screen != "Не указано":
        parts.append(f"Область: {screen}.")

    if target_columns:
        parts.append(f"Целевые элементы: {', '.join(_limit_items(target_columns, 8))}.")

    if technical_plan:
        parts.append(f"Основа решения: {_inline_list(technical_plan, limit=3)}.")

    if checks:
        parts.append(f"Критерий готовности: {_inline_list(checks, limit=2)}.")

    if restrictions:
        parts.append(f"Ограничения: {_inline_list(restrictions, limit=2)}.")

    if missing_inputs:
        parts.append(f"Пока не хватает: {', '.join(_limit_items(missing_inputs, 4))}.")
    elif known_inputs:
        parts.append(f"Опора: {', '.join(_limit_items(known_inputs, 4))}.")

    return " ".join(parts)


def _build_as_artifacts(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    az_fix_plan = (
        current_result.get("az_fix_plan")
        if isinstance(current_result.get("az_fix_plan"), dict)
        else {}
    )

    title = _normalize_text(payload.get("title")) or _normalize_text(az_fix_plan.get("title")) or "Рабочий артефакт"
    screen = _normalize_text(payload.get("screen")) or _normalize_text(az_fix_plan.get("screen")) or "Не указано"
    target_columns = _normalize_str_list(payload.get("target_columns")) or _normalize_str_list(
        az_fix_plan.get("target_columns")
    )
    task_type = _normalize_text(task.get("task_type")) or "unknown"

    implementation_steps: list[dict[str, Any]] = []
    for idx, step in enumerate(az_fix_plan.get("technical_plan") or [], start=1):
        implementation_steps.append(
            {
                "step_no": idx,
                "title_ru": f"Шаг {idx}",
                "description_ru": str(step),
            }
        )

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
        artifact_text_lines.extend([f"- {column}" for column in target_columns])
    else:
        artifact_text_lines.append("- (не указаны)")

    artifact_text_lines.extend(["", "План реализации (из brief AZ):"])
    for item in implementation_steps:
        artifact_text_lines.append(f"{item['step_no']}. {item['description_ru']}")

    checks = _normalize_str_list(
        az_fix_plan.get("post_fix_checks") or payload.get("acceptance_criteria")
    )

    artifact_text_lines.extend(["", "Проверки после исправления:"])
    if checks:
        artifact_text_lines.extend([f"- {check}" for check in checks])
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
            "Следующий этап - AK проверяет полноту, риски и соответствие brief/чек-листу.",
        ],
    }


def _as_handoff_allowed(task: dict[str, Any]) -> tuple[bool, str]:
    target_agent = (task.get("target_agent") or "").upper()
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    next_agent = (result.get("next_agent") or "").upper()
    handoff_ready = bool(result.get("handoff_ready"))
    task_status = (task.get("status") or "").upper()
    next_action = (result.get("next_action") or "").lower()

    if target_agent == "AS":
        return True, ""

    if task_status == "BRIEF_READY" and handoff_ready and next_agent == "AS":
        return True, ""

    if task_status == "REVIEW_NEEDS_ATTENTION" and next_action in {"return_to_as", "ak_return_to_as"}:
        return True, ""

    return (
        False,
        (
            "AS не может принять задачу: ожидается target_agent='AS', "
            "или handoff от AZ (status=BRIEF_READY, handoff_ready=true, next_agent='AS'), "
            "или возврат после AK (status=REVIEW_NEEDS_ATTENTION, next_action=return_to_as)."
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
        final_answer = _build_final_answer(task_id, task)

        prev_result = task.get("result") if isinstance(task.get("result"), dict) else {}
        prev_result = prev_result if isinstance(prev_result, dict) else {}

        current_review_cycle = int(prev_result.get("review_cycle") or 0)
        is_rework = (prev_result.get("next_action") or "").lower() in {"return_to_as", "ak_return_to_as"}
        if is_rework:
            current_review_cycle += 1

        merged_result = {
            **prev_result,
            "as_executor": "AS",
            "as_artifacts": as_artifacts,
            "as_output": {
                "answer_text": final_answer,
            },
            "final_answer": final_answer,
            "as_status": "artifacts_ready",
            "handoff_ready": True,
            "next_agent": "AK",
            "as_completed_at": _now_iso(),
            "next_action": "as_artifacts_ready",
            "review_cycle": current_review_cycle,
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
                "review_cycle": current_review_cycle,
                "has_final_answer": bool(final_answer),
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
                "review_cycle": current_review_cycle,
                "has_final_answer": bool(final_answer),
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
                    "review_cycle": current_review_cycle,
                    "has_final_answer": bool(final_answer),
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
