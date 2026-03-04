from __future__ import annotations

import ast
import json
import operator as op
import os
import re
import base64
import io
import csv
import subprocess
import tempfile
import uuid
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

_ATT_MAX_IMAGES = 5
_ATT_MAX_IMAGE_BYTES_EACH = 2_900_000
_ATT_MAX_TOTAL_IMAGE_BYTES = 2_900_000


_VIDEO_EXTS = {"mp4", "mov", "mkv", "avi", "webm", "m4v"}
_VIDEO_FPS = float(os.getenv("OZONATOR_VIDEO_FPS", "1") or 1)  # 1 кадр/сек
_VIDEO_BATCH_SIZE = int(os.getenv("OZONATOR_VIDEO_BATCH_SIZE", "8") or 8)  # кадров на 1 vision-вызов
_VIDEO_MAX_FRAMES = int(os.getenv("OZONATOR_VIDEO_MAX_FRAMES", "900") or 900)  # защита от очень длинных видео
_VIDEO_FRAME_DIM = int(os.getenv("OZONATOR_VIDEO_FRAME_DIM", "1024") or 1024)

_VIDEO_AUDIO_MAX_BYTES = int(os.getenv("OZONATOR_VIDEO_AUDIO_MAX_BYTES", "24000000") or 24000000)
_ASR_MODEL = (os.getenv("OZONATOR_ASR_MODEL") or "whisper-large-v3-turbo").strip()
_ASR_LANGUAGE = (os.getenv("OZONATOR_ASR_LANGUAGE") or "ru").strip()

_VIDEO_MAX_TRANSCRIPT_CHARS = int(os.getenv("OZONATOR_VIDEO_MAX_TRANSCRIPT_CHARS", "12000") or 12000)
_VIDEO_MAX_TIMELINE_LINES = int(os.getenv("OZONATOR_VIDEO_MAX_TIMELINE_LINES", "1200") or 1200)


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




# Groq: base64 encoded image request max is 4MB (base64 payload)
# (см. Groq Vision docs)
_MAX_GROQ_BASE64_IMAGE_CHARS = 4 * 1024 * 1024


def _prepare_image_for_vision(file_name: str, content_type: str, raw: bytes) -> tuple[bytes, str]:
    """Подготовка изображения для vision: размер, формат, совместимость."""
    mime = _guess_image_mime(file_name, content_type)
    if not raw:
        return raw, mime

    def b64_len(n: int) -> int:
        return ((n + 2) // 3) * 4

    try:
        from PIL import Image
    except Exception:
        return raw, mime

    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        max_dim = int(os.getenv("OZONATOR_MAX_IMAGE_DIM", "2048") or 2048)
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        target_b64 = int(os.getenv("OZONATOR_MAX_IMAGE_B64", str(_MAX_GROQ_BASE64_IMAGE_CHARS)) or _MAX_GROQ_BASE64_IMAGE_CHARS)
        quality = 90

        def encode(q: int) -> bytes:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            return buf.getvalue()

        data = encode(quality)
        while b64_len(len(data)) > target_b64 and quality >= 45:
            quality -= 7
            data = encode(quality)

        attempts = 0
        while b64_len(len(data)) > target_b64 and attempts < 3:
            w, h = img.size
            img = img.resize((max(1, int(w * 0.85)), max(1, int(h * 0.85))))
            data = encode(max(45, quality))
            attempts += 1

        if b64_len(len(data)) <= target_b64:
            return data, "image/jpeg"

        return raw, mime
    except Exception:
        return raw, mime


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _probe_video_duration_seconds(path: str) -> float:
    exe = _ffmpeg_exe()
    try:
        p = subprocess.run([exe, "-hide_banner", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        m = _DURATION_RE.search(p.stderr.decode("utf-8", "ignore"))
        if not m:
            return 0.0
        h = int(m.group(1))
        mi = int(m.group(2))
        s = float(m.group(3))
        return h * 3600 + mi * 60 + s
    except Exception:
        return 0.0


def _multipart_encode(fields: dict[str, str], files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----ozonator" + uuid.uuid4().hex
    body = io.BytesIO()

    for k, v in (fields or {}).items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.write((v or "").encode("utf-8"))
        body.write(b"\r\n")

    for field, filename, ctype, data in files:
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
        body.write(data or b"")
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def _audio_transcriptions_url(chat_url: str) -> str:
    u = (chat_url or "").strip()
    if "/chat/completions" in u:
        return u.rsplit("/chat/completions", 1)[0] + "/audio/transcriptions"
    return u.rstrip("/") + "/audio/transcriptions"


def _asr_transcribe(audio_bytes: bytes, mime: str, filename: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    payload = payload or {}
    key = _pick_llm_key()
    if not key or not audio_bytes:
        return None

    try:
        import urllib.request as urllib_request
        import urllib.error as urllib_error

        chat_url = _normalize_llm_url(os.getenv("OPENAI_API_URL", ""), key=key)
        url = _audio_transcriptions_url(chat_url)

        fields = {"model": _ASR_MODEL, "response_format": "verbose_json", "temperature": "0"}
        if _ASR_LANGUAGE:
            fields["language"] = _ASR_LANGUAGE

        body, ctype = _multipart_encode(fields, [("file", filename, mime, audio_bytes)])

        req = urllib_request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", ctype)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "Mozilla/5.0 OzonatorAgents-AS")

        with urllib_request.urlopen(req, timeout=120) as r:
            raw = r.read()
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


def _format_asr_verbose(resp: dict[str, Any] | None) -> str:
    if not isinstance(resp, dict):
        return ""
    segs = resp.get("segments")
    if isinstance(segs, list) and segs:
        out = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            t0 = float(s.get("start") or 0.0)
            t1 = float(s.get("end") or 0.0)
            txt = str(s.get("text") or "").strip()
            if not txt:
                continue

            def fmt(t: float) -> str:
                mm = int(t // 60)
                ss = int(t % 60)
                return f"{mm:02d}:{ss:02d}"

            out.append(f"[{fmt(t0)}–{fmt(t1)}] {txt}")
        return "\n".join(out).strip()
    txt = str(resp.get("text") or "").strip()
    return txt


def _extract_audio_bytes(video_path: str) -> tuple[bytes, str, str]:
    exe = _ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if os.path.exists(wav_path):
            data = open(wav_path, "rb").read()
            if len(data) <= _VIDEO_AUDIO_MAX_BYTES:
                return data, "audio/wav", "audio.wav"

        mp3_path = os.path.join(td, "audio.mp3")
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", mp3_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if os.path.exists(mp3_path):
            data = open(mp3_path, "rb").read()
            return data, "audio/mpeg", "audio.mp3"

    return b"", "", ""


def _extract_frames_1fps(video_path: str) -> list[tuple[int, bytes, str]]:
    exe = _ffmpeg_exe()
    frames: list[tuple[int, bytes, str]] = []
    with tempfile.TemporaryDirectory() as td:
        out_pattern = os.path.join(td, "frame_%06d.jpg")
        vf = f"fps={_VIDEO_FPS},scale='min({_VIDEO_FRAME_DIM},iw)':-2"
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-vf", vf, "-q:v", "4", out_pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        names = sorted([n for n in os.listdir(td) if n.startswith("frame_") and n.lower().endswith(".jpg")])
        for idx, name in enumerate(names):
            t_sec = int(idx / _VIDEO_FPS) if _VIDEO_FPS else idx
            raw = open(os.path.join(td, name), "rb").read()
            processed, mime = _prepare_image_for_vision(name, "image/jpeg", raw)
            frames.append((t_sec, processed, mime))
            if len(frames) >= _VIDEO_MAX_FRAMES:
                break
    return frames


def _llm_chat_raw(messages: list[dict[str, Any]], *, model: str, url: str, key: str, max_tokens: int = 900, temperature: float = 0.2) -> str:
    try:
        import urllib.request as urllib_request
        import urllib.error as urllib_error

        body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 OzonatorAgents-AS"
                ),
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=120) as r:
            raw = r.read()
        resp = json.loads(raw.decode("utf-8", "replace"))
        return _extract_chat_completion_text(resp)
    except Exception:
        return ""


def _vision_timeline_batch(frames: list[tuple[int, bytes, str]], *, memory: str, payload: dict[str, Any] | None = None) -> tuple[list[str], str]:
    payload = payload or {}
    key = _pick_llm_key()
    if not key or not frames:
        return [], memory

    chat_url = _normalize_llm_url(os.getenv("OPENAI_API_URL", ""), key=key)
    model = _pick_llm_model(payload)

    # Если есть изображения — выбираем актуальную vision-модель (у Groq старые llama-3.2-*-vision-preview уже сняты с поддержки).
    vision_model = (os.getenv("OZONATOR_LLM_VISION_MODEL") or os.getenv("GROQ_VISION_MODEL") or "").strip()
    if key.startswith("gsk_"):
        v_low = vision_model.strip().lower()
        if (not vision_model) or (v_low in _DEPRECATED_GROQ_VISION_MODELS):
            model = _DEFAULT_GROQ_VISION_MODEL
        else:
            model = vision_model
    else:
        if vision_model:
            model = vision_model

    times = ", ".join([f"{t}s" for t, _, _ in frames])
    prompt = (
        "Это последовательные кадры ОДНОГО видео, строго по времени (1 кадр = 1 секунда). "
        f"Временные метки кадров по порядку: {times}.\n"
        f"Сквозной контекст до этого (если пусто — это начало): {memory or 'нет'}.\n\n"
        "Сделай:\n"
        "1) Для каждого кадра — ОДНА строка строго формата: t=<сек>: <что видно/что происходит>.\n"
        "2) В конце ОДНА строка: MEMORY: <обновлённый сквозной контекст для следующих кадров: кто/где/что происходит, что меняется>.\n"
        "Не выдумывай, пиши только по кадрам."
    )

    image_parts = []
    for _t, data, mime in frames:
        b64 = base64.b64encode(data).decode("ascii")
        image_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}})

    system_msg = "Ты — Екатерина. Отвечай на русском, коротко и по делу. Не обсуждай тему ИИ."
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": [{"type": "text", "text": prompt}, *image_parts]}]

    text = _llm_chat_raw(messages, model=model, url=chat_url, key=key, max_tokens=1200, temperature=0.2).strip()
    if not text:
        return [], memory

    new_memory = memory
    if "MEMORY:" in text:
        before, after = text.split("MEMORY:", 1)
        new_memory = after.strip()
        lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    return lines, new_memory


def _analyze_video_to_block(file_name: str, content_type: str, content: bytes, *, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    if not content:
        return f"— {file_name}: (видео пустое)"

    with tempfile.TemporaryDirectory() as td:
        ext = _ext(file_name) or "mp4"
        video_path = os.path.join(td, f"video.{ext}")
        with open(video_path, "wb") as f:
            f.write(content)

        dur = _probe_video_duration_seconds(video_path)

        audio_bytes, audio_mime, audio_fn = _extract_audio_bytes(video_path)
        transcript = ""
        if audio_bytes:
            asr = _asr_transcribe(audio_bytes, audio_mime, audio_fn, payload=payload)
            transcript = _format_asr_verbose(asr)

        frames = _extract_frames_1fps(video_path)
        memory = ""
        timeline: list[str] = []

        for i in range(0, len(frames), _VIDEO_BATCH_SIZE):
            batch = frames[i : i + _VIDEO_BATCH_SIZE]
            lines, memory = _vision_timeline_batch(batch, memory=memory, payload=payload)
            timeline.extend(lines)

        if len(timeline) > _VIDEO_MAX_TIMELINE_LINES:
            timeline = timeline[:_VIDEO_MAX_TIMELINE_LINES] + ["…(таймлайн обрезан по лимиту строк)"]

        transcript = _trim_text(transcript, _VIDEO_MAX_TRANSCRIPT_CHARS)
        memory = _trim_text(memory, 3000)

        parts = [f"— {file_name} (видео)"]
        if dur:
            parts.append(f"Длительность: ~{int(dur)} сек.")
        if transcript:
            parts.append("Речь/звук (транскрипция):\n" + transcript)
        if timeline:
            parts.append("Кадры (1 кадр/сек):\n" + "\n".join(timeline))
        if memory:
            parts.append("Сквозной контекст (для сопоставления кадров):\n" + memory)

        return "\n".join(parts).strip()



def _collect_attachments_for_llm(task_id: int, payload: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """
    Возвращает:
    1) Текстовый блок-превью вложений (таблицы/текст и пометки о пропусках).
    2) Список image_url частей (OpenAI-compatible) для vision-моделей.
    """
    payload = payload or {}

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

            # Сначала ужимаем/уменьшаем, затем применяем лимиты уже к обработанным данным.
            processed, mime = _prepare_image_for_vision(file_name, content_type, content)
            processed_size = len(processed or b"")

            if processed_size > _ATT_MAX_IMAGE_BYTES_EACH:
                blocks.append(
                    f"— {file_name}: (изображение, пропущено — даже после подготовки слишком большой файл {processed_size} bytes, лимит {_ATT_MAX_IMAGE_BYTES_EACH})"
                )
                continue
            if total_image_bytes + processed_size > _ATT_MAX_TOTAL_IMAGE_BYTES:
                blocks.append(
                    f"— {file_name}: (изображение, пропущено — достигнут лимит по суммарному размеру изображений {_ATT_MAX_TOTAL_IMAGE_BYTES} bytes)"
                )
                continue

            b64 = base64.b64encode(processed).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            image_parts.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
            total_image_bytes += processed_size
            blocks.append(f"— {file_name}: (изображение, приложено к сообщению)")
            continue

        # --- Video (frames 1fps + audio transcription) ---
        if ext in _VIDEO_EXTS or (content_type or "").lower().startswith("video/"):
            try:
                blocks.append(_analyze_video_to_block(file_name, content_type, content, payload=payload))
            except Exception:
                blocks.append(f"— {file_name}: (видео, не удалось разобрать)")
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
# Output post-processing: enforce feminine first-person (RU)
# -------------------------

def _case_like(src: str, dst: str) -> str:
    if not src:
        return dst
    return dst[:1].upper() + dst[1:] if src[:1].isupper() else dst


# Только базовые и самые частые формы. Правим ТОЛЬКО 1-е лицо или начало фразы.
_FEM_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bя\s+рад\b", re.I), "я рада"),
    (re.compile(r"(^|[\.\!\?\n]\s*)(рад)(\s+за\b)", re.I), r"\1рада\3"),
    (re.compile(r"\bя\s+сделал\b", re.I), "я сделала"),
    (re.compile(r"\bя\s+понял\b", re.I), "я поняла"),
    (re.compile(r"\bя\s+увидел\b", re.I), "я увидела"),
    (re.compile(r"\bя\s+не\s+увидел\b", re.I), "я не увидела"),
    (re.compile(r"\bя\s+получил\b", re.I), "я получила"),
    (re.compile(r"\bя\s+посмотрел\b", re.I), "я посмотрела"),
    (re.compile(r"\bя\s+прочитал\b", re.I), "я прочитала"),
    (re.compile(r"\bя\s+спросил\b", re.I), "я спросила"),
    (re.compile(r"\bя\s+готов\b", re.I), "я готова"),
    (re.compile(r"\bя\s+уверен\b", re.I), "я уверена"),
    (re.compile(r"\bя\s+должен\b", re.I), "я должна"),
    (re.compile(r"\bя\s+смог\b", re.I), "я смогла"),
]


def _enforce_feminine_ru(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    out = text

    # Начало сообщения: "Рад..." -> "Рада..."
    out = re.sub(
        r"(^|[\r\n]+)\s*(Рад)(\b)",
        lambda m: (m.group(1) or "") + _case_like(m.group(2), "Рада"),
        out,
    )

    for pat, repl in _FEM_RULES:
        def _sub(m: re.Match[str]) -> str:
            # если repl содержит групповые ссылки, не трогаем кейс
            if "\\" in repl:
                return m.expand(repl)
            src = m.group(0)
            if src[:1].isupper():
                return repl[:1].upper() + repl[1:]
            return repl

        out = pat.sub(_sub, out)

    return out


def _safe_strip(s: str) -> str:
    return (s or "").strip()

# -------------------------
# LLM (Groq / OpenAI-compatible Chat Completions)
# -------------------------
_DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Альтернативный vision‑модель (может быть доступна/устаревать; используем только как fallback)
_DEFAULT_GROQ_VISION_MODEL_FALLBACK = "meta-llama/llama-4-maverick-17b-128e-instruct"
_DEPRECATED_GROQ_VISION_MODELS = {
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llava-v1.5-7b-4096-preview",
}


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
        # Если прилетели изображения — выбираем актуальную vision-модель (у Groq старые llama-3.2-*-vision-preview уже сняты с поддержки).
        vision_model = (os.getenv("OZONATOR_LLM_VISION_MODEL") or os.getenv("GROQ_VISION_MODEL") or "").strip()
        if key.startswith("gsk_"):
            v_low = vision_model.strip().lower()
            if (not vision_model) or (v_low in _DEPRECATED_GROQ_VISION_MODELS):
                model = _DEFAULT_GROQ_VISION_MODEL
            else:
                model = vision_model
        else:
            # Для OpenAI/других OpenAI‑compatible провайдеров — используем то, что задано пользователем.
            if vision_model:
                model = vision_model

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
        f"Ты — Екатерина. Говори от первого лица и о себе ВСЕГДА в женском роде (я сделала, я посмотрела, я готова). НИКОГДА не используй формы мужского рода про себя. Мы с пользователем давно знакомы, общение по-свойски. "
        f"Пользователя зовут {user_name}. Всегда обращайся к нему на {pronoun}. "
        f"Используй обращение «{addr}» (можно переключаться по контексту между: {', '.join(variants)}). "
        "Не представляйся и не объясняй, кто ты. "
        f"{forbid_ai_line} "
        "Отвечай на русском, кратко, точно и по делу. "
        "Восклицательные знаки используй только при крайней необходимости; в письмах «!» воспринимается как крик. "
        "Всегда опирайся ТОЛЬКО на факты из последнего сообщения пользователя и переданной истории диалога. "
        "НЕ гадай и НЕ перечисляй возможные трактовки. "
        "Если во входном сообщении есть изображения — они переданы тебе, анализируй их. НИКОГДА не утверждай, что ты их не видишь. НИКОГДА не делай предположений о личности людей на фото (например, что это пользователь), если он сам это не сказал. "
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
            vision_text = (
                "Проанализируй изображение(я) и ответь на вопрос пользователя. "
                "Сначала зафиксируй наблюдения: (1) люди/лица (есть ли человек, сколько, где; если не уверена — так и скажи), "
                "(2) основные объекты/сцена, (3) текст/надписи. "
                "Не выдумывай детали и не делай предположений о личности людей на фото. "
                "Верни ответ СТРОГО в JSON-объекте с ключами: people_present (true/false/unknown), people, objects, text, answer. "
                f"\nВопрос пользователя: {question_ru}"
            )
            user_content = [{"type": "text", "text": vision_text}, *image_parts]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": question_ru})
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }

        # Groq vision поддерживает JSON mode (response_format)
        if image_parts and key.startswith("gsk_"):
            body["response_format"] = {"type": "json_object"}

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
            if status == 403:
                return f"{addr}, у меня ошибка: запрос к провайдеру заблокирован (403 Forbidden)."
            if status == 401:
                return f"{addr}, у меня ошибка: ключ провайдера отклонён (401 Unauthorized)."

            # Пытаемся вытащить человекочитаемое сообщение (без секретов)
            try:
                err_raw = e.read()  # type: ignore[attr-defined]
                if err_raw:
                    err_data = json.loads(err_raw.decode("utf-8", errors="ignore"))
                    if isinstance(err_data, dict):
                        err_obj = err_data.get("error") if isinstance(err_data.get("error"), dict) else {}
                        msg = err_obj.get("message") if isinstance(err_obj.get("message"), str) else ""
                        if msg.strip():
                            msg = msg.strip()
                            if len(msg) > 240:
                                msg = msg[:240] + "…"
                            low_msg = msg.lower()
                            # Если модель была снята с поддержки — пробуем автоматически переключиться на актуальную vision‑модель Groq.
                            if image_parts and key.startswith("gsk_") and ("decommissioned" in low_msg or "no longer supported" in low_msg):
                                try:
                                    body2 = dict(body)
                                    body2["model"] = _DEFAULT_GROQ_VISION_MODEL
                                    req2 = urllib_request.Request(
                                        url,
                                        data=json.dumps(body2, ensure_ascii=False).encode("utf-8"),
                                        headers=req.headers,
                                        method="POST",
                                    )
                                    with urllib_request.urlopen(req2, timeout=60) as r2:
                                        raw2 = r2.read()
                                    data2 = json.loads(raw2.decode("utf-8")) if raw2 else {}
                                    ans2 = _extract_chat_completion_text(data2)
                                    if isinstance(ans2, str) and ans2.strip():
                                        return _enforce_feminine_ru(_strip_and_apply_self_profile_updates(ans2.strip()))
                                except Exception:
                                    pass

                            return f"{addr}, я получила ошибку от провайдера: {msg}"
            except Exception:
                pass

            return ""

        data = json.loads(raw.decode("utf-8")) if raw else {}
        ans_text = _safe_strip(_extract_chat_completion_text(data))

        # Если мы просили JSON (vision) — распарсим и вернём человеческий answer
        if image_parts:
            obj = None
            try:
                if ans_text.startswith("{"):
                    obj = json.loads(ans_text)
            except Exception:
                obj = None

            if isinstance(obj, dict):
                people_present = obj.get("people_present")
                people = _safe_strip(str(obj.get("people") or ""))
                objects = _safe_strip(str(obj.get("objects") or ""))
                txt = _safe_strip(str(obj.get("text") or ""))
                answer = _safe_strip(str(obj.get("answer") or ""))

                ql = (question_ru or "").lower()
                need_people_check = any(k in ql for k in [
                    "кто на фото", "кто на снимке", "есть ли человек", "человек", "лицо", "люди", "персона",
                    "что на фото", "что на снимке", "что изображено",
                ])

                if need_people_check and str(people_present).lower() in {"false", "0", "no", "none", "unknown", ""}:
                    try:
                        vision_text2 = (
                            "Проверь ТОЛЬКО наличие людей/частей тела/лица на изображении(ях). "
                            "Если человек есть даже частично — скажи об этом. "
                            "Верни СТРОГО JSON: people_present (true/false/unknown), people."
                        )
                        messages2 = [{"role": "system", "content": system_msg}]
                        messages2.extend(history)
                        messages2.append({
                            "role": "user",
                            "content": [{"type": "text", "text": vision_text2}, *image_parts],
                        })
                        body2 = {
                            "model": model,
                            "messages": messages2,
                            "temperature": 0.1,
                            "max_tokens": 300,
                            "response_format": {"type": "json_object"},
                        }
                        req2 = urllib_request.Request(
                            url,
                            data=json.dumps(body2, ensure_ascii=False).encode("utf-8"),
                            headers=req.headers,
                            method="POST",
                        )
                        with urllib_request.urlopen(req2, timeout=60) as r2:
                            raw2 = r2.read()
                        data2 = json.loads(raw2.decode("utf-8")) if raw2 else {}
                        ans2 = _safe_strip(_extract_chat_completion_text(data2))
                        obj2 = json.loads(ans2) if ans2.startswith("{") else None
                        if isinstance(obj2, dict) and str(obj2.get("people_present")).lower() in {"true", "1", "yes"}:
                            people_present = obj2.get("people_present")
                            people = _safe_strip(str(obj2.get("people") or people))
                    except Exception:
                        pass

                if not answer:
                    parts = []
                    if str(people_present).lower() in {"true", "1", "yes"} and people:
                        parts.append(f"Люди: {people}.")
                    elif str(people_present).lower() in {"false", "0", "no"}:
                        parts.append("Людей на фото не вижу уверенно.")
                    elif str(people_present).lower() == "unknown":
                        parts.append("По людям на фото не уверена (качество/ракурс).")
                    if objects:
                        parts.append(f"Объекты/сцена: {objects}.")
                    if txt:
                        parts.append(f"Текст на изображении: {txt}.")
                    answer = " ".join(parts).strip()

                ans_text = answer or ans_text

        ans_text = _strip_and_apply_self_profile_updates(ans_text)
        ans_text = _enforce_feminine_ru(ans_text)
        return ans_text
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

    attachments_text, image_parts = _collect_attachments_for_llm(task_id, payload)
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
        return _enforce_feminine_ru(ai_answer)

    if image_parts:
        return (
            "Саша, я получила фото, но текущая настройка модели на сервере не принимает изображения. "
            "Поставь в Render для сервиса AS переменную OZONATOR_LLM_VISION_MODEL "
            "= meta-llama/llama-4-scout-17b-16e-instruct и перезапусти AS. "
            "После этого снова отправь фото — я смогу его разобрать."
        )

    heur = _heuristic_answer(main_goal)
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
