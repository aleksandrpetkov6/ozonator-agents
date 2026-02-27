from __future__ import annotations

import json
import os
import re
import ast
import operator as op
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

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

# ============================================================
# Ozonator Agents AS
# Purpose (MVP):
# - Receive handoff from AZ (BRIEF_READY) or rework from AK.
# - Build artifacts bundle (mock for now).
# - Produce a USER-FACING final_answer.
#
# Patch v2:
# - Add "direct answer" mode for generic user questions.
#   1) If OpenAI key is configured on server: call Responses API and return answer.
#   2) Else: try small built-in heuristics (math/unit conversions/FAQ).
#   3) Else: fallback to previous template ("Готово... артефакты...").
# ============================================================

app = FastAPI(title="Ozonator Agents AS")


# -------------------------
# Helpers
# -------------------------
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


# -------------------------
# Direct answering (optional)
# -------------------------
# IMPORTANT:
# - Секреты (API keys) должны храниться ТОЛЬКО в env сервиса (Render), никогда в клиенте.
# - Groq использует OpenAI-совместимый Chat Completions.
# - Код нормализует URL до рабочего /chat/completions и НЕ логирует секреты.
#
# Поддерживаемые переменные окружения (на сервисе AS):
#   GROQ_API_KEY  (рекомендуется)
#   OPENAI_API_KEY (fallback)
#   Agents        (legacy, как на вашем скрине)
# Опционально:
#   OPENAI_MODEL или GROQ_MODEL (например: llama-3.3-70b-versatile)
#   OPENAI_API_URL (можно указывать базу /v1 или полный /chat/completions)

_DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _pick_llm_key(provider: str) -> str:
    provider = (provider or "").strip().lower()
    # Prefer explicit provider keys, then fallbacks.
    if provider == "openai":
        return (os.getenv("OPENAI_API_KEY") or os.getenv("Agents") or "").strip()
    # default: groq
    return (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("Agents") or "").strip()


def _pick_llm_model(payload: dict) -> str:
    # Priority: payload -> GROQ_MODEL/OPENAI_MODEL -> default Groq model
    m = str((payload or {}).get("llm_model") or "").strip()
    if m:
        return m
    return (os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


def _pick_llm_provider(payload: dict) -> str:
    p = str((payload or {}).get("llm_provider") or "").strip().lower()
    if p:
        return p
    return (os.getenv("LLM_PROVIDER") or "groq").strip().lower() or "groq"


def _normalize_llm_url(raw_url: str, provider: str) -> str:
    url = (raw_url or "").strip()
    provider = (provider or "").strip().lower() or "groq"

    if not url:
        return _DEFAULT_OPENAI_URL if provider == "openai" else _DEFAULT_GROQ_URL

    low = url.lower().rstrip("/")

    # If user passed a base like ".../v1" or ".../openai/v1" → append /chat/completions
    if low.endswith("/v1") or low.endswith("/openai/v1"):
        return url.rstrip("/") + "/chat/completions"

    # If points to /responses → rewrite to /chat/completions (Groq doesn't support Responses API)
    if "/responses" in low and "/chat/completions" not in low:
        return re.sub(r"/responses.*$", "/chat/completions", url, flags=re.IGNORECASE)

    # If contains provider base but missing /chat/completions
    if ("groq.com" in low or "api.openai.com" in low) and ("/chat/completions" not in low):
        return url.rstrip("/") + "/chat/completions"

    return url


def _llm_chat_complete(question_ru: str, *, provider: str, model: str, url: str, key: str) -> tuple[str, dict]:
    # Returns (answer_text, diag). diag is safe (no secrets).
    if not key:
        return "", {"ok": False, "error": "missing_api_key", "provider": provider}

    system_msg = (
        "Ты — ассистент в приложении Ozonator Agents. "
        "Отвечай на русском. Коротко и по делу. "
        "Не упоминай внутренние статусы/агентов/оркестрацию."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question_ru},
        ],
        "temperature": 0.2,
        # Groq/OpenAI-compatible параметр. Добавляем, чтобы избежать 400 на некоторых прокси/обвязках.
        "max_tokens": 512,
    }

    try:
        import urllib.request as urllib_request
        import urllib.error as urllib_error

        req = urllib_request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=60) as r:
                raw = r.read()
                status = getattr(r, "status", 200)
        except urllib_error.HTTPError as e:
            # Read body to understand root cause (still safe — no secrets).
            status = int(getattr(e, "code", 0) or 0)
            raw = b""
            try:
                raw = e.read() or b""
            except Exception:
                raw = b""

            data = {}
            err_msg = ""
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                err_msg = raw.decode("utf-8", errors="replace")[:500] if raw else ""

            if isinstance(data, dict):
                # OpenAI-style error
                err = data.get("error")
                if isinstance(err, dict):
                    err_msg = str(err.get("message") or err.get("type") or "")[:500]
                elif isinstance(data.get("detail"), str):
                    err_msg = str(data.get("detail"))[:500]

            return "", {
                "ok": False,
                "error": "HTTPError",
                "http_status": status,
                "provider": provider,
                "model": model,
                "url": url,
                "message": err_msg,
            }

        data = json.loads(raw.decode("utf-8")) if raw else {}

        # OpenAI-compatible Chat Completions parsing
        text = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message") if isinstance(first.get("message"), dict) else {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                text = content.strip()

        # Иногда обвязки кладут ответ в `choices[0].text`
        if not text and isinstance(choices, list) and choices and isinstance(choices[0], dict):
            t = choices[0].get("text")
            if isinstance(t, str) and t.strip():
                text = t.strip()

        return text, {"ok": bool(text), "http_status": int(status or 0), "provider": provider, "model": model, "url": url}

    except Exception as e:
        kind = e.__class__.__name__
        http_status = None
        if hasattr(e, "code"):
            try:
                http_status = int(getattr(e, "code") or 0)
            except Exception:
                http_status = None
        return "", {"ok": False, "error": kind, "http_status": http_status, "provider": provider, "model": model, "url": url, "message": str(e)[:300]}

def _llm_answer_task(task: dict[str, Any], question_ru: str) -> tuple[str, dict]:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    provider = _pick_llm_provider(payload)
    model = _pick_llm_model(payload)
    env_url = os.getenv("OPENAI_API_URL", "")
    url = _normalize_llm_url(env_url, provider)
    key = _pick_llm_key(provider)

    return _llm_chat_complete(question_ru, provider=provider, model=model, url=url, key=key)


# Backward compatible alias (old name used by some code)
def _openai_answer(question_ru: str) -> str:
    provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower() or "groq"
    model = (os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
    url = _normalize_llm_url(os.getenv("OPENAI_API_URL", ""), provider)
    key = _pick_llm_key(provider)
    text, _diag = _llm_chat_complete(question_ru, provider=provider, model=model, url=url, key=key)
    return text


# -------------------------
# Heuristic fallback (no external AI)
# -------------------------
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval_arith(expr: str) -> Optional[float]:
    """
    Safe arithmetic evaluator for expressions like "2+2*3".
    Allows digits, (), + - * / // % ** and spaces.
    """
    expr = expr.strip()
    if not expr:
        return None

    # Quick allowlist
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)%]*", expr):
        return None

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in __ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand)))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return float(_ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right)))
        raise ValueError("Unsupported expression")

    try:
        return _eval(node)
    except Exception:
        return None


def _heuristic_answer(question: str) -> str:
    q = question.strip()
    if not q:
        return ""

    q_low = q.lower()

    # Unit conversion: cm in meter
    if ("сантиметр" in q_low or "см" in q_low) and ("метр" in q_low or "м" in q_low) and "сколько" in q_low:
        # Typical: "сколько сантиметров в метре?"
        return "100 сантиметров."

    # Nearest star to Earth
    if ("ближайш" in q_low and "земл" in q_low and "звезд" in q_low):
        return "Солнце. Если исключить Солнце — Проксима Центавра."

    # Arithmetic: "сколько будет 2+2" or any question that is mostly math
    m = re.search(r"(?:сколько\s+будет|сколько)\s*([0-9\.\s\+\-\*\/\(\)%]+)\??", q_low)
    if m:
        val = _safe_eval_arith(m.group(1))
        if val is not None:
            # Pretty format: int if whole
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return str(val)

    # Also handle "2+2?" (pure expression)
    if re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)%]+\??", q_low):
        expr = q_low.replace("?", "").strip()
        val = _safe_eval_arith(expr)
        if val is not None:
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return str(val)

    return ""


# -------------------------
# Core logic
# -------------------------
def _build_final_answer(task_id: int, task: dict[str, Any]) -> str:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    az_fix_plan = (
        current_result.get("az_fix_plan") if isinstance(current_result.get("az_fix_plan"), dict) else {}
    )

    user_request = _normalize_text(payload.get("user_request"))
    payload_brief = _normalize_text(payload.get("brief"))
    goal = _normalize_text(az_fix_plan.get("goal"))
    title = _normalize_text(payload.get("title")) or _normalize_text(az_fix_plan.get("title"))
    screen = _normalize_text(payload.get("screen")) or _normalize_text(az_fix_plan.get("screen"))
    target_columns = _normalize_str_list(payload.get("target_columns")) or _normalize_str_list(
        az_fix_plan.get("target_columns")
    )
    main_goal = user_request or payload_brief or goal or title or f"задача #{task_id}"

    endpoints = _normalize_str_list(
        payload.get("endpoints")
        or payload.get("endpoint")
        or az_fix_plan.get("endpoints")
        or az_fix_plan.get("endpoint")
    )
    request_keys = _normalize_str_list(
        payload.get("request_keys")
        or payload.get("input_keys")
        or az_fix_plan.get("request_keys")
        or az_fix_plan.get("input_keys")
    )
    response_keys = _normalize_str_list(
        payload.get("response_keys")
        or payload.get("output_keys")
        or az_fix_plan.get("response_keys")
        or az_fix_plan.get("output_keys")
    )
    link_keys = _normalize_str_list(
        payload.get("link_keys")
        or payload.get("binding_keys")
        or az_fix_plan.get("link_keys")
        or az_fix_plan.get("binding_keys")
    )

    goal_lower = main_goal.lower()
    is_api_mapping_task = bool(endpoints or request_keys or response_keys or link_keys) or (
        "api" in goal_lower and "ozon" in goal_lower
    )
    if is_api_mapping_task:
        endpoint_text = ", ".join(endpoints) if endpoints else "не переданы"
        request_text = ", ".join(request_keys) if request_keys else "не переданы"
        response_text = ", ".join(response_keys) if response_keys else "не переданы"
        link_text = ", ".join(link_keys) if link_keys else "не переданы"
        return " ".join(
            [
                f"Endpoint: {endpoint_text}.",
                f"Ключи запроса: {request_text}.",
                f"Ключи ответа: {response_text}.",
                f"Связка между вызовами: {link_text}.",
            ]
        )
    # 1) LLM (server-side) if configured
    ai_answer, ai_diag = _llm_answer_task(task, main_goal)
    if ai_answer:
        return ai_answer

    # 2) Heuristic fallback (no external AI)
    heur = _heuristic_answer(main_goal)
    if heur:
        return heur

    # 3) Если LLM не ответил — вернуть понятную причину и 1 следующий шаг
    if isinstance(ai_diag, dict) and not ai_diag.get('ok'):
        err = str(ai_diag.get('error') or '').strip()
        code = ai_diag.get('http_status')
        msg = str(ai_diag.get('message') or '').strip()
        provider = str(ai_diag.get('provider') or '').strip() or 'groq'
        model = str(ai_diag.get('model') or '').strip()

        # Универсальная подсказка для сервиса AS
        common_next = (
            "Следующий шаг: в Render → ozonator-as-dev → Environment проверь ключ (GROQ_API_KEY или Agents) "
            "и что OPENAI_API_URL пустой или равен https://api.groq.com/openai/v1/chat/completions; затем дождись деплоя."
        )

        if err == 'missing_api_key':
            return (
                "Ошибка: на сервере AS не настроен ключ LLM (нет GROQ_API_KEY/OPENAI_API_KEY/Agents). "
                "" + common_next
            )

        if code in (401, 403):
            return (
                "Ошибка: LLM отклонил запрос (unauthorized). "
                "" + common_next
            )

        if code == 404:
            return (
                "Ошибка: неверный LLM URL (HTTP 404). "
                "" + common_next
            )

        if code == 429:
            return (
                "Ошибка: лимит/перегрузка LLM (HTTP 429). "
                "Следующий шаг: повтори запрос позже или временно смени модель на llama-3.1-8b-instant."
            )

        if code == 400:
            extra = f" ({msg})" if msg else ""
            return (
                f"Ошибка: LLM вернул HTTP 400{extra}. "
                "Следующий шаг: проверь модель (llm_model) и endpoint; для Groq модель должна быть из списка GroqDocs (например llama-3.3-70b-versatile)."
            )

        # Таймауты/сетевые ошибки
        if err.lower() in {'timeouterror', 'sockettimeout', 'urlerror'} or (code is None and err):
            extra = f" ({err}: {msg})" if msg else f" ({err})"
            return (
                f"Ошибка: LLM не ответил (сеть/таймаут){extra}. "
                "Следующий шаг: проверь в Render, что сервис AS имеет доступ в интернет и повтори запрос."
            )

        extra = f" ({err}{': ' + msg if msg else ''})" if err or msg else ""
        return (
            f"Ошибка: LLM не дал ответ{extra}. "
            + common_next
        )

    # 4) Previous template fallback
    parts = [f"Готово. Подготовлено решение по задаче: {main_goal}."]
    if screen and screen != "Не указано":
        parts.append(f"Область: {screen}.")
    if target_columns:
        parts.append(f"Целевые элементы: {', '.join(target_columns)}.")
    parts.append("Артефакты собраны и переданы на проверку AK.")
    return " ".join(parts)


def _build_as_artifacts(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    az_fix_plan = (
        current_result.get("az_fix_plan") if isinstance(current_result.get("az_fix_plan"), dict) else {}
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
            {"step_no": idx, "title_ru": f"Шаг {idx}", "description_ru": str(step)}
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

    checks = _normalize_str_list(az_fix_plan.get("post_fix_checks") or payload.get("acceptance_criteria"))
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


def _as_handoff_allowed(task: dict[str, Any]) -> Tuple[bool, str]:
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
        "AS не может принять задачу: ожидается target_agent='AS', "
        "или handoff от AZ (status=BRIEF_READY, handoff_ready=true, next_agent='AS'), "
        "или возврат после AK (status=REVIEW_NEEDS_ATTENTION, next_action=return_to_as).",
    )


# -------------------------
# Base / Health
# -------------------------
@app.get("/")
def root():
    return {"service": "AS", "status": "ok", "message": "Ozonator Agents AS service is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "AS"}


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, detail = check_postgres(settings.database_url)
    return {"service": "AS", "component": "db", "ok": ok, "detail": detail}


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, detail = check_redis(settings.redis_url)
    return {"service": "AS", "component": "redis", "ok": ok, "detail": detail}


@app.get("/health/all")
def health_all():
    settings = get_settings()
    ok_db, db_detail = check_postgres(settings.database_url)
    ok_redis, redis_detail = check_redis(settings.redis_url)
    return {
        "service": "AS",
        "status": "ok" if (ok_db and ok_redis) else "degraded",
        "components": {"db": {"ok": ok_db, "detail": db_detail}, "redis": {"ok": ok_redis, "detail": redis_detail}},
    }


# -------------------------
# Tasks read endpoints
# -------------------------
@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={"service": "AS", "operation": "get_task", "status": "error", "message": message, "task": None},
        )
    return JSONResponse(status_code=200, content={"service": "AS", "operation": "get_task", "status": "ok", "message": "OK", "task": task})


@app.get("/tasks/{task_id}/logs")
def get_task_logs_endpoint(task_id: int):
    settings = get_settings()
    ok, _task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={"service": "AS", "operation": "get_task_logs", "status": "error", "message": message, "task_id": task_id, "logs": None},
        )
    ok, logs, message = get_task_logs(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"service": "AS", "operation": "get_task_logs", "status": "error", "message": message, "task_id": task_id, "logs": None},
        )
    return JSONResponse(
        status_code=200,
        content={"service": "AS", "operation": "get_task_logs", "status": "ok", "message": "OK", "task_id": task_id, "count": len(logs), "logs": logs},
    )


# -------------------------
# AS runner
# -------------------------
@app.post("/as/run-task/{task_id}")
def as_run_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={"service": "AS", "operation": "as_run_task", "status": "error", "message": message, "task": None},
        )

    allowed, deny_message = _as_handoff_allowed(task)
    if not allowed:
        return JSONResponse(
            status_code=400,
            content={"service": "AS", "operation": "as_run_task", "status": "error", "message": deny_message, "task": task},
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AS",
        event_type="task_run_started",
        level="info",
        message="AS начал сборку артефактов",
        meta={"source_agent": task.get("source_agent"), "task_type": task.get("task_type"), "prev_status": task.get("status")},
    )

    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"service": "AS", "operation": "as_run_task", "status": "error", "message": message, "task": None},
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
            "as_output": {"answer_text": final_answer},
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

        ok_set, task, message_set = set_task_result(settings.database_url, task_id=task_id, result=merged_result, error_message=None)
        if not ok_set:
            return JSONResponse(
                status_code=503,
                content={"service": "AS", "operation": "as_run_task", "status": "error", "message": message_set, "task": None},
            )

        ok, task, message = update_task_status(settings.database_url, task_id, "ARTIFACTS_READY")
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"service": "AS", "operation": "as_run_task", "status": "error", "message": message, "task": None},
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
        # Do not leak secrets.
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
            meta={"error": "AS error"},
        )
        return JSONResponse(
            status_code=500,
            content={"service": "AS", "operation": "as_run_task", "status": "error", "message": err, "task": None},
        )
