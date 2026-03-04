from __future__ import annotations

import ast
import json
import operator as op
import os
import re
import base64
import io
import csv
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import psycopg
import redis
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
from db.files import get_task_file_content, list_task_files

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


_ATT_MAX_FILES = 8
_ATT_MAX_TOTAL_BYTES = 2 * 1024 * 1024
_ATT_MAX_CHARS_PER_FILE = 20000

_ATT_MAX_IMAGES = 4
_ATT_MAX_IMAGE_BYTES_EACH = 2 * 1024 * 1024
_ATT_MAX_TOTAL_IMAGE_BYTES = 4 * 1024 * 1024


def _ext(name: str) -> str:
    name = (name or "").lower()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def _decode_text(raw: bytes) -> str:
    if raw is None:
        return ""
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            return raw.decode("cp1251")
        except Exception:
            return raw.decode("utf-8", errors="replace")


def _trim_text(s: str, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n…(обрезано)"


def _preview_csv(text: str, delimiter: str, max_rows: int = 60) -> str:
    if not text:
        return "(пусто)"
    try:
        f = io.StringIO(text)
        reader = csv.reader(f, delimiter=delimiter)
        out_lines = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                out_lines.append("…(обрезано)")
                break
            out_lines.append("\t".join(row))
        return "\n".join(out_lines).strip() or "(пусто)"
    except Exception:
        return _trim_text(text, _ATT_MAX_CHARS_PER_FILE)


def _preview_xlsx(raw: bytes) -> str:
    try:
        from openpyxl import load_workbook
    except Exception:
        return "(xlsx: openpyxl не установлен)"

    try:
        wb = load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
        ws = wb.worksheets[0] if wb.worksheets else None
        if ws is None:
            return "(xlsx: нет листов)"

        max_rows = 60
        max_cols = 25
        lines = []
        for r_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=max_rows, max_col=max_cols, values_only=True), start=1):
            vals = []
            for v in row:
                if v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            lines.append("\t".join(vals).rstrip())
        txt = "\n".join(lines).strip()
        return txt or "(xlsx: пусто)"
    except Exception:
        return "(xlsx: не удалось прочитать)"


def _guess_image_mime(file_name: str, content_type: str | None = None) -> str:
    ct = (content_type or "").strip().lower()
    if ct.startswith("image/"):
        return ct
    ext = _ext(file_name)
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    if ext == "gif":
        return "image/gif"
    return "image/jpeg"


def _collect_attachments_for_llm(task_id: int) -> tuple[str, list[dict[str, Any]]]:
    """
    Возвращает:
    1) Текстовый блок-превью вложений (таблицы/текст и пометки о пропусках).
    2) Список image_url частей (OpenAI-compatible) для vision-моделей.
    """
    settings = get_settings()
    db_url = getattr(settings, "database_url", None)
    if not db_url:
        return "", []

    ok, files, _message = list_task_files(db_url, task_id)
    if not ok or not files:
        return "", []

    total_text_bytes = 0
    total_image_bytes = 0
    image_parts: list[dict[str, Any]] = []
    blocks: list[str] = []

    for meta in files[:_ATT_MAX_FILES]:
        file_id = int(meta.get("id") or 0)
        file_name = str(meta.get("file_name") or f"file_{file_id}")
        size_bytes = int(meta.get("size_bytes") or 0)
        content_type = str(meta.get("content_type") or "")

        ok_c, _meta_c, content, _msg_c = get_task_file_content(db_url, task_id, file_id)
        if not ok_c or content is None:
            blocks.append(f"— {file_name}: (не удалось загрузить содержимое)")
            continue

        ext = _ext(file_name)

        # --- Images (vision) ---
        if ext in {"jpg", "jpeg", "png", "webp", "gif"} or (content_type or "").lower().startswith("image/"):
            if len(image_parts) >= _ATT_MAX_IMAGES:
                blocks.append(f"— {file_name}: (изображение, пропущено — достигнут лимит {_ATT_MAX_IMAGES} шт.)")
                continue
            if size_bytes > _ATT_MAX_IMAGE_BYTES_EACH:
                blocks.append(
                    f"— {file_name}: (изображение, пропущено — слишком большой файл {size_bytes} bytes, лимит {_ATT_MAX_IMAGE_BYTES_EACH})"
                )
                continue
            if total_image_bytes + size_bytes > _ATT_MAX_TOTAL_IMAGE_BYTES:
                blocks.append(
                    f"— {file_name}: (изображение, пропущено — достигнут лимит по суммарному размеру изображений {_ATT_MAX_TOTAL_IMAGE_BYTES} bytes)"
                )
                continue

            mime = _guess_image_mime(file_name, content_type)
            b64 = base64.b64encode(content).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            total_image_bytes += size_bytes
            blocks.append(f"— {file_name}: (изображение, приложено к сообщению)")
            continue

        # --- Text-like ---
        if total_text_bytes + size_bytes > _ATT_MAX_TOTAL_BYTES:
            blocks.append("(достигнут лимит по суммарному размеру текстовых вложений, остальное пропущено)")
            break

        total_text_bytes += len(content)
        if ext in {"txt", "md", "log", "json", "yaml", "yml"}:
            txt = _trim_text(_decode_text(content), _ATT_MAX_CHARS_PER_FILE)
            blocks.append(f"— {file_name}\n{txt}")
            continue
        if ext == "csv":
            txt = _decode_text(content)
            blocks.append(f"— {file_name}\n{_preview_csv(txt, ',', 60)}")
            continue
        if ext == "tsv":
            txt = _decode_text(content)
            blocks.append(f"— {file_name}\n{_preview_csv(txt, '\t', 60)}")
            continue
        if ext in {"xlsx", "xlsm"}:
            blocks.append(f"— {file_name}\n{_preview_xlsx(content)}")
            continue

        blocks.append(f"— {file_name}: (бинарный формат, автоматический разбор не поддерживается)")

    if not blocks and not image_parts:
        return "", []

    text_block = "\n\n".join(["Вложения к задаче (используй их при ответе):", *blocks]).strip() if blocks else ""
    return text_block, image_parts


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _normalize_conversation_history(
    payload: dict[str, Any] | None,
    *,
    max_items: int = 24,
    max_chars: int = 7000,
) -> list[dict[str, str]]:
    """
    payload.conversation_history = [{"role":"user|assistant","content":"..."}]
    Урезаем историю, чтобы ускорять ответы.
    """
    if not isinstance(payload, dict):
        return []

    raw = payload.get("conversation_history") or payload.get("conversation") or []
    if not isinstance(raw, list):
        return []

    prepared: list[dict[str, str]] = []
    total = 0

    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if not content:
            continue
        if len(content) > 1400:
            content = content[:1400]
        if prepared and total + len(content) > max_chars:
            break

        prepared.append({"role": role, "content": content})
        total += len(content)

        if len(prepared) >= max_items:
            break

    return prepared


def _pick_addressing(user_text: str, default_name: str, variants: list[str]) -> str:
    """
    Выбираем обращение к Александру по контексту. Это только тон, не влияет на смысл.
    """
    t = (user_text or "").lower()

    formal = any(
        k in t
        for k in [
            "договор",
            "претенз",
            "официаль",
            "коммерческ",
            "письмо",
            "юрид",
            "счет",
            "счёт",
            "акт",
            "инвойс",
        ]
    )
    if formal and "Александр Николаевич" in variants:
        return "Александр Николаевич"

    tech = any(
        k in t
        for k in [
            "репо",
            "github",
            "коммит",
            "ветк",
            "pull",
            "pr",
            "issue",
            "лог",
            "ошиб",
            "stack",
            "trace",
            "yaml",
            "json",
            "api",
            "endpoint",
            "render",
            "swagger",
            "sql",
            "postgres",
            "redis",
            "ui",
            "tkinter",
            "python",
            "pip",
        ]
    )
    if tech and "Александр" in variants:
        return "Александр"

    warm = any(
        k in t for k in ["устал", "пережива", "бесит", "задолб", "нерв", "обид", "поддерж", "плохо", "тяжело"]
    )
    if warm and "Сашечка" in variants:
        return "Сашечка"

    return default_name or (variants[0] if variants else "Саша")


# -------------------------
# Екатерина: SelfProfile (persisted)
# -------------------------
_SELF_PROFILE_REDIS_KEY = "aa:ekaterina:self_profile:v2"
_SELF_PROFILE_DB_AGENT_CODE = "EKATERINA"
_SELF_PROFILE_DB_STATE_KEY = "self_profile_v2"

_SELF_PROFILE_UPDATE_RE = re.compile(
    r"\[\[EK_SELF_PROFILE_UPDATE\]\](.*?)\[\[/EK_SELF_PROFILE_UPDATE\]\]",
    re.S,
)

_REDIS_CLIENT: redis.Redis | None = None


def _get_redis_client() -> redis.Redis | None:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    settings = get_settings()
    if not getattr(settings, "redis_url", None):
        return None

    try:
        _REDIS_CLIENT = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        return _REDIS_CLIENT
    except Exception:
        _REDIS_CLIENT = None
        return None


def _default_self_profile() -> dict[str, Any]:
    return {
        "name": "Екатерина",
        "hair_color": "каштановые",
        "eyes": "ярко-голубые",
        "vibe": "по-свойски, спокойная, без услужливости",
    }


def _db_load_self_profile() -> dict[str, Any] | None:
    settings = get_settings()
    db_url = getattr(settings, "database_url", None)
    if not db_url:
        return None
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT state_json
                    FROM agent_state
                    WHERE agent_code = %s AND state_key = %s
                    LIMIT 1;
                    """,
                    (_SELF_PROFILE_DB_AGENT_CODE, _SELF_PROFILE_DB_STATE_KEY),
                )
                row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip():
            try:
                obj = json.loads(val)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None
    except Exception:
        return None


def _db_save_self_profile(profile: dict[str, Any]) -> None:
    settings = get_settings()
    db_url = getattr(settings, "database_url", None)
    if not db_url:
        return
    try:
        payload_json = json.dumps(profile, ensure_ascii=False)
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_state (agent_code, state_key, state_json, updated_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (agent_code, state_key)
                    DO UPDATE SET state_json = EXCLUDED.state_json, updated_at = NOW();
                    """,
                    (_SELF_PROFILE_DB_AGENT_CODE, _SELF_PROFILE_DB_STATE_KEY, payload_json),
                )
            conn.commit()
    except Exception:
        return


def _load_self_profile() -> dict[str, Any]:
    r = _get_redis_client()
    if r is not None:
        try:
            raw = r.get(_SELF_PROFILE_REDIS_KEY)
            if raw:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

    prof = _db_load_self_profile()
    if isinstance(prof, dict) and prof:
        if r is not None:
            try:
                r.set(_SELF_PROFILE_REDIS_KEY, json.dumps(prof, ensure_ascii=False))
            except Exception:
                pass
        return prof

    prof = _default_self_profile()
    if r is not None:
        try:
            r.set(_SELF_PROFILE_REDIS_KEY, json.dumps(prof, ensure_ascii=False))
        except Exception:
            pass
    _db_save_self_profile(prof)
    return prof


def _save_self_profile(profile: dict[str, Any]) -> None:
    r = _get_redis_client()
    if r is not None:
        try:
            r.set(_SELF_PROFILE_REDIS_KEY, json.dumps(profile, ensure_ascii=False))
        except Exception:
            pass
    _db_save_self_profile(profile)


def _apply_self_profile_update(update_obj: dict[str, Any]) -> None:
    try:
        current = _load_self_profile()
        set_part = update_obj.get("set") if isinstance(update_obj.get("set"), dict) else {}
        del_part = update_obj.get("delete") if isinstance(update_obj.get("delete"), list) else []

        for k, v in set_part.items():
            if isinstance(k, str) and k.strip():
                current[k.strip()] = v
        for k in del_part:
            if isinstance(k, str) and k.strip():
                current.pop(k.strip(), None)

        _save_self_profile(current)
    except Exception:
        return


def _strip_and_apply_self_profile_updates(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    cleaned = text
    for m in list(_SELF_PROFILE_UPDATE_RE.finditer(text)):
        payload = (m.group(1) or "").strip()
        try:
            upd = json.loads(payload)
            if isinstance(upd, dict):
                _apply_self_profile_update(upd)
        except Exception:
            pass
        cleaned = cleaned.replace(m.group(0), "")

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# -------------------------
# LLM (Groq / OpenAI-compatible Chat Completions)
# -------------------------
_DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_OPENAI_VISION_MODEL = "gpt-4o-mini"


def _clean_api_key(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    s = "".join(ch for ch in s if 33 <= ord(ch) <= 126)
    return s

def _mask_secrets_in_text(s: str) -> str:
    """
    Убирает возможные секреты из диагностических сообщений.
    Важно: НЕ логируем и НЕ возвращаем ключи.
    """
    if not s:
        return ""
    s = re.sub(r"gsk_[A-Za-z0-9_\-]{10,}", "gsk_***", s)
    s = re.sub(r"sk-[A-Za-z0-9]{10,}", "sk-***", s)
    s = re.sub(r"(?i)bearer\s+[A-Za-z0-9_\-]{10,}", "Bearer ***", s)
    return s


def _pick_llm_key() -> str:
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "Agents"):
        key = _clean_api_key(os.getenv(name, ""))
        if key:
            return key
    return ""


def _pick_llm_model(payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    m = str(payload.get("llm_model") or "").strip()
    if m:
        return m
    return (os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or _DEFAULT_GROQ_MODEL).strip() or _DEFAULT_GROQ_MODEL


def _normalize_llm_url(raw_url: str, *, key: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        base = _DEFAULT_GROQ_BASE if key.startswith("gsk_") else _DEFAULT_OPENAI_BASE
        return base + "/chat/completions"

    u = url.rstrip("/")
    low = u.lower()

    if low.endswith("/v1") or low.endswith("/openai/v1"):
        return u + "/chat/completions"

    if ("groq.com" in low or "openai.com" in low) and ("/chat/completions" not in low) and ("/responses" not in low):
        return u + "/chat/completions"

    return u


def _is_chat_completions(url: str) -> bool:
    return "/chat/completions" in (url or "").lower()


def _extract_chat_completion_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else None
        if first:
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for c in content:
                        if isinstance(c, str) and c.strip():
                            parts.append(c.strip())
                        elif isinstance(c, dict):
                            text = c.get("text") or c.get("content")
                            if isinstance(text, str) and text.strip():
                                parts.append(text.strip())
                    if parts:
                        return "\n".join(parts).strip()
    return ""


def _llm_answer(question_ru: str, payload: dict[str, Any] | None = None, *, image_parts: list[dict[str, Any]] | None = None) -> str:
    key = _pick_llm_key()
    if not key:
        return ""

    payload = payload or {}
    url = _normalize_llm_url(os.getenv("OPENAI_API_URL", ""), key=key)
    model = _pick_llm_model(payload)

    image_parts = image_parts or []
    if image_parts:
        # Если прилетели изображения — пытаемся выбрать vision-модель.
        vision_model = (
            os.getenv("OZONATOR_LLM_VISION_MODEL")
            or os.getenv("GROQ_VISION_MODEL")
            or os.getenv("OPENAI_VISION_MODEL")
            or ""
        ).strip()
        if vision_model:
            model = vision_model
        else:
            low_m = (model or "").lower()
            if key.startswith("gsk_"):
                # Groq: автоматически выбираем vision-модель
                if ("vision" not in low_m and "image" not in low_m):
                    model = "llama-3.2-11b-vision-preview"
            else:
                # OpenAI/compatible: если выбрана текстовая модель — переключаемся на vision по умолчанию
                if ("gpt-4o" not in low_m) and ("vision" not in low_m) and ("image" not in low_m):
                    model = _DEFAULT_OPENAI_VISION_MODEL

    prefs = payload.get("user_prefs") if isinstance(payload.get("user_prefs"), dict) else {}
    user_name = str(prefs.get("user_name") or "Александр").strip() or "Александр"
    variants = prefs.get("addressing_variants")
    if not isinstance(variants, list) or not variants:
        variants = ["Александр", "Саша", "Сашечка", "Александр Николаевич"]
    variants = [str(x).strip() for x in variants if str(x).strip()]
    default_addr = str(prefs.get("addressing_default") or "Саша").strip() or "Саша"
    addr = _pick_addressing(question_ru, default_addr, variants)
    pronoun = str(prefs.get("pronoun") or "ты").strip() or "ты"
    never_ai = bool(prefs.get("never_discuss_ai", True))

    try:
        prof = _load_self_profile()
    except Exception:
        prof = _default_self_profile()

    forbid_ai_line = (
        f"ВАЖНО: не обсуждай и не упоминай тему ИИ/нейросетей/моделей вообще. Даже если {user_name} это упомянул — не подхватывай тему."
        if never_ai
        else ""
    )

    system_msg = (
        f"Ты — Екатерина. Говори от первого лица и о себе всегда в женском роде (я сделала, я посмотрела, я готова). Мы с пользователем давно знакомы, общение по-свойски. "
        f"Пользователя зовут {user_name}. Всегда обращайся к нему на {pronoun}. "
        f"Используй обращение «{addr}» (можно переключаться по контексту между: {', '.join(variants)}). "
        "Не представляйся и не объясняй, кто ты. "
        f"{forbid_ai_line} "
        "Отвечай на русском, кратко, точно и по делу. "
        "Восклицательные знаки используй только при крайней необходимости; в письмах «!» воспринимается как крик. "
        "Всегда опирайся ТОЛЬКО на факты из последнего сообщения пользователя и переданной истории диалога. "
        "НЕ гадай и НЕ перечисляй возможные трактовки. "
        "Если во входном сообщении есть изображения — они переданы тебе, анализируй их и не утверждай, что ты их не видишь. "
        "Если по контексту нельзя понять, что требуется (или не хватает критичных данных), задай ОДИН уточняющий вопрос и остановись. "
        "Твой SelfProfile (внутренний образ) должен быть стабильным и сохраняться между сообщениями и рестартами. "
        "Не рассказывай о себе без повода. "
        "Если в ответе всё же прозвучал ЛЮБОЙ факт о тебе — зафиксируй его скрытым update-блоком в самом конце ответа без пояснений: "
        "[[EK_SELF_PROFILE_UPDATE]]{\"set\":{},\"delete\":[]}[[/EK_SELF_PROFILE_UPDATE]] "
        f"SelfProfile(JSON): {json.dumps(prof, ensure_ascii=False)}"
    )

    history = _normalize_conversation_history(payload)

    try:
        import urllib.error as urllib_error
        import urllib.request as urllib_request

        if not _is_chat_completions(url):
            return ""

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history)
        if image_parts:
            user_content = [{"type": "text", "text": question_ru}, *image_parts]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": question_ru})

        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500,
        }

        req = urllib_request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36 OzonatorAgents-AS"
                ),
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except urllib_error.HTTPError as e:
            status = int(getattr(e, "code", 0) or 0)
            err_raw = b""
            try:
                err_raw = e.read()  # type: ignore[attr-defined]
            except Exception:
                err_raw = b""

            if status == 403:
                return "Ошибка: запрос к провайдеру заблокирован (403 Forbidden)."
            if status == 401:
                return "Ошибка: ключ отклонён (401 Unauthorized)."

            # Пытаемся вытащить человекочитаемое сообщение (без секретов)
            try:
                if err_raw:
                    err_data = json.loads(err_raw.decode("utf-8", errors="ignore"))
                    if isinstance(err_data, dict):
                        err_obj = err_data.get("error") if isinstance(err_data.get("error"), dict) else {}
                        msg = err_obj.get("message") if isinstance(err_obj.get("message"), str) else ""
                        if msg.strip():
                            msg = _mask_secrets_in_text(msg.strip())
                            if len(msg) > 240:
                                msg = msg[:240] + "…"
                            return f"Ошибка: {msg}"
            except Exception:
                pass

            # Фолбэк: короткий снэпшот тела ошибки
            try:
                if err_raw:
                    snippet = err_raw.decode("utf-8", errors="ignore").strip()
                    snippet = re.sub(r"\s+", " ", snippet)
                    snippet = _mask_secrets_in_text(snippet)
                    if snippet:
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "…"
                        return f"Ошибка HTTP {status}: {snippet}"
            except Exception:
                pass

            return f"Ошибка HTTP {status} при обращении к провайдеру модели."

        data = json.loads(raw.decode("utf-8")) if raw else {}
        return _strip_and_apply_self_profile_updates(_extract_chat_completion_text(data))
    except Exception:
        return ""


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
    expr = expr.strip()
    if not expr:
        return None
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
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
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

    m = re.search(r"(?:сколько\s+будет|сколько)\s*([0-9\.\s\+\-\*\/\(\)%]+)\??", q_low)
    if m:
        val = _safe_eval_arith(m.group(1))
        if val is not None:
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return str(val)

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
    az_fix_plan = current_result.get("az_fix_plan") if isinstance(current_result.get("az_fix_plan"), dict) else {}

    user_request = _normalize_text(payload.get("user_request"))
    payload_brief = _normalize_text(payload.get("brief"))
    goal = _normalize_text(az_fix_plan.get("goal"))
    title = _normalize_text(payload.get("title")) or _normalize_text(az_fix_plan.get("title"))
    screen = _normalize_text(payload.get("screen")) or _normalize_text(az_fix_plan.get("screen"))
    target_columns = _normalize_str_list(payload.get("target_columns")) or _normalize_str_list(az_fix_plan.get("target_columns"))

    main_goal = user_request or payload_brief or goal or title or f"задача #{task_id}"

    attachments_text, image_parts = _collect_attachments_for_llm(task_id)
    llm_question = (main_goal + "\n\n" + attachments_text).strip() if attachments_text else main_goal

    endpoints = _normalize_str_list(payload.get("endpoints") or payload.get("endpoint") or az_fix_plan.get("endpoints") or az_fix_plan.get("endpoint"))
    request_keys = _normalize_str_list(payload.get("request_keys") or payload.get("input_keys") or az_fix_plan.get("request_keys") or az_fix_plan.get("input_keys"))
    response_keys = _normalize_str_list(payload.get("response_keys") or payload.get("output_keys") or az_fix_plan.get("response_keys") or az_fix_plan.get("output_keys"))
    link_keys = _normalize_str_list(payload.get("link_keys") or payload.get("binding_keys") or az_fix_plan.get("link_keys") or az_fix_plan.get("binding_keys"))

    goal_lower = main_goal.lower()
    is_api_mapping_task = bool(endpoints or request_keys or response_keys or link_keys) or ("api" in goal_lower and "ozon" in goal_lower)

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

    ai_answer = _llm_answer(llm_question, payload, image_parts=image_parts)
    if ai_answer:
        return ai_answer
    if image_parts:
        key = _pick_llm_key()
        if not key:
            return (
                "Саша, я получила фото, но на сервере сейчас не настроен ключ для обращения к провайдеру модели. "
                "Поэтому я отвечаю в офлайн-режиме и картинки разобрать не могу. "
                "Один шаг: в Render → сервис AS → Environment добавь GROQ_API_KEY (предпочтительно) "
                "или OPENAI_API_KEY и перезапусти AS. После этого снова отправь фото — я разберу его."
            )

        probe = _llm_answer("Ответь одним словом: ОК.", payload)
        if not probe or probe.strip().lower().startswith("ошибка"):
            # Если текстовый вызов тоже не проходит — это не про vision, а про доступ/ключ/URL
            base_hint = (
                "GROQ_API_KEY (предпочтительно)" if key.startswith("gsk_") else "OPENAI_API_KEY"
            )
            return (
                "Саша, я получила фото, но сейчас сервер не может нормально обратиться к провайдеру модели (ключ/URL/доступ). "
                f"Один шаг: в Render → сервис AS → Environment проверь {base_hint} (и при необходимости OPENAI_API_URL), "
                "затем перезапусти AS. После этого снова отправь фото."
            )

        provider = "Groq" if key.startswith("gsk_") else "OpenAI/compatible"
        rec_model = "llama-3.2-11b-vision-preview" if key.startswith("gsk_") else _DEFAULT_OPENAI_VISION_MODEL
        return (
            f"Саша, я получила фото, но провайдер {provider} не принял изображение в текущей настройке модели. "
            f"Один шаг: в Render → сервис AS → Environment задай OZONATOR_LLM_VISION_MODEL = {rec_model} и перезапусти AS. "
            "После этого снова отправь фото — я разберу его по содержимому."
        )

    heur = _heuristic_answer(main_goal)(main_goal)
    if heur:
        return heur

    parts = [f"Готово. Я подготовила решение по задаче: {main_goal}."]
    if screen and screen != "Не указано":
        parts.append(f"Область: {screen}.")
    if target_columns:
        parts.append(f"Целевые элементы: {', '.join(target_columns)}.")
    parts.append("Я собрала артефакты и передала их на проверку AK.")
    return " ".join(parts)


def _build_as_artifacts(task_id: int, task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    az_fix_plan = current_result.get("az_fix_plan") if isinstance(current_result.get("az_fix_plan"), dict) else {}

    title = _normalize_text(payload.get("title")) or _normalize_text(az_fix_plan.get("title")) or "Рабочий артефакт"
    screen = _normalize_text(payload.get("screen")) or _normalize_text(az_fix_plan.get("screen")) or "Не указано"
    target_columns = _normalize_str_list(payload.get("target_columns")) or _normalize_str_list(az_fix_plan.get("target_columns"))
    task_type = _normalize_text(task.get("task_type")) or "unknown"

    implementation_steps: list[dict[str, Any]] = []
    for idx, step in enumerate(az_fix_plan.get("technical_plan") or [], start=1):
        implementation_steps.append({"step_no": idx, "title_ru": f"Шаг {idx}", "description_ru": str(step)})

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
# Health
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
