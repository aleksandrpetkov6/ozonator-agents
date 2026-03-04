# -*- coding: utf-8 -*-
"""
Ozonator Agents Client (.pyw) — single file, no external deps (Tkinter + stdlib).

Назначение:
- Одно окно: пишешь задачу -> создаётся task в AA -> запускается оркестрация -> показывается task.result.final_answer.
- Быстрый отклик: сразу показываем "…" (как печатает), даже если сеть/LLM дольше.

AA API (основные пути):
- POST /tasks/create
- POST /aa/run-task/{task_id}
- GET  /tasks/{task_id}
- GET  /tasks/{task_id}/logs

ENV:
- OZONATOR_AA_BASE_URL (default: https://ozonator-aa-dev.onrender.com)
- OZONATOR_AA_BEARER (Authorization: Bearer ...)
- OZONATOR_AA_ADMIN_TOKEN (X-Admin-Token ...)
"""

import json
import io
import os
import hashlib
import threading
import time
import tkinter as tk
import uuid
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib import parse as urllib_parse


DEFAULT_AA_BASE_URL = (os.getenv("OZONATOR_AA_BASE_URL") or "https://ozonator-aa-dev.onrender.com").rstrip("/")
AA_BEARER = (os.getenv("OZONATOR_AA_BEARER") or "").strip()
AA_ADMIN_TOKEN = (os.getenv("OZONATOR_AA_ADMIN_TOKEN") or "").strip()

DEFAULT_LLM_PROVIDER = (os.getenv("OZONATOR_LLM_PROVIDER") or "groq").strip()
DEFAULT_LLM_MODEL = (os.getenv("OZONATOR_LLM_MODEL") or "llama-3.3-70b-versatile").strip()

CREATE_RETRIES = 2
HTTP_TIMEOUT = int(os.getenv("OZONATOR_HTTP_TIMEOUT") or 90)
UPLOAD_TIMEOUT = int(os.getenv("OZONATOR_UPLOAD_TIMEOUT") or max(180, HTTP_TIMEOUT))
UPLOAD_RETRIES = int(os.getenv("OZONATOR_UPLOAD_RETRIES") or 2)
SEND_TIMEOUT_SEC = 75  # ожидание получения task_id (create_task)
FAST_POLL_MS = 250
NORMAL_POLL_MS = 1000
FAST_POLL_WINDOW_SEC = 12



# Радикально иной подход к вложениям: по умолчанию отправляем не оригинал фото,
# а "превью" (ресайз + JPEG), чтобы не ловить write timeout на медленных каналах.
# Оригинал при этом остаётся локально у пользователя.
CLIENT_IMAGE_MAX_BYTES = int(os.getenv("OZONATOR_CLIENT_IMAGE_MAX_BYTES") or 900_000)  # ~0.9MB
CLIENT_IMAGE_MAX_DIM = int(os.getenv("OZONATOR_CLIENT_IMAGE_MAX_DIM") or 1600)
CLIENT_IMAGE_PREVIEW_ONLY = (os.getenv("OZONATOR_CLIENT_IMAGE_PREVIEW_ONLY") or "1").strip().lower() not in ("0", "false", "no")

try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False


STATE_VERSION = 1
STATE_MAX_ITEMS = max(50, int(os.getenv("OZONATOR_STATE_MAX_ITEMS") or 200))
STATE_FILE_NAME = "chat_state_v1.json"
CONFIG_FILE_NAME = "client_config_v1.json"


def _app_data_dir() -> str:
    """Папка для пользовательских данных клиента.

    В Windows обычно переживает переустановку программы (если не чистить профиль).
    """
    try:
        if os.name == "nt":
            base = os.getenv("APPDATA") or os.path.expanduser("~")
            path = os.path.join(base, "OzonatorAgents")
        else:
            base = os.getenv("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
            path = os.path.join(base, "ozonator_agents")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        # fallback
        path = os.path.join(os.path.expanduser("~"), "ozonator_agents")
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        return path


def _read_json_file(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_atomic(path: str, data: dict) -> None:
    """Атомарная запись, чтобы не убить историю при внезапном закрытии."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _load_or_create_user_key(config_path: str, bearer: str, admin_token: str) -> str:
    """Стабильный ключ пользователя для восстановления истории.

    Приоритет:
    1) OZONATOR_USER_KEY (если задан)
    2) sha256 от OZONATOR_AA_BEARER / OZONATOR_AA_ADMIN_TOKEN (ключи не светим)
    3) сохранённый в config (uuid)
    """
    env_key = (os.getenv("OZONATOR_USER_KEY") or "").strip()
    if env_key:
        return env_key

    basis = (bearer or "").strip() or (admin_token or "").strip()
    if basis:
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    cfg = _read_json_file(config_path) or {}
    key = str(cfg.get("user_key") or "").strip()
    if key:
        return key

    key = uuid.uuid4().hex
    _write_json_atomic(config_path, {"version": 1, "user_key": key})
    return key


def _is_image_name(file_name: str, content_type: str) -> bool:
    ct = (content_type or "").lower().strip()
    if ct.startswith("image/"):
        return True
    ext = (file_name or "").lower().rsplit(".", 1)
    ext = ext[-1] if len(ext) == 2 else ""
    return ext in {"jpg", "jpeg", "png", "webp", "gif"}


def _make_image_preview(raw: bytes, max_dim: int, max_bytes: int):
    """Делает JPEG-превью <= max_bytes. Требует Pillow. Возвращает (bytes, content_type) или (None, None)."""
    if not PIL_AVAILABLE or Image is None:
        return (None, None)
    try:
        img = Image.open(io.BytesIO(raw))
        if getattr(img, "mode", "") == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif getattr(img, "mode", "") != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        def encode(q: int) -> bytes:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            return buf.getvalue()

        q = 82
        data = encode(q)
        while len(data) > max_bytes and q >= 45:
            q -= 7
            data = encode(q)

        tries = 0
        while len(data) > max_bytes and tries < 3:
            w, h = img.size
            img = img.resize((max(1, int(w * 0.85)), max(1, int(h * 0.85))))
            data = encode(max(45, q))
            tries += 1

        if len(data) <= max_bytes:
            return (data, "image/jpeg")
        return (None, None)
    except Exception:
        return (None, None)


def _prepare_file_for_upload(file_path: str):
    """Возвращает (upload_name, content_type, content_bytes, note)."""
    file_name = os.path.basename(file_path) or "file"
    ctype = __import__('mimetypes').guess_type(file_name)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        raw = f.read() or b""

    if _is_image_name(file_name, ctype):
        need_preview = CLIENT_IMAGE_PREVIEW_ONLY or (len(raw) > CLIENT_IMAGE_MAX_BYTES)
        if need_preview:
            data, new_ct = _make_image_preview(raw, CLIENT_IMAGE_MAX_DIM, CLIENT_IMAGE_MAX_BYTES)
            if data is not None and new_ct is not None:
                base = file_name.rsplit(".", 1)[0]
                upload_name = f"{base}_preview.jpg"
                note = f"отправлено превью ({len(data)} bytes)"
                return upload_name, new_ct, data, note
            note = "превью не получилось (нет Pillow или ошибка), отправлен оригинал"
            return file_name, ctype, raw, note

    return file_name, ctype, raw, "отправлен оригинал"
HISTORY_MAX_ITEMS = 24

AA_DISPLAY_NAME = "Екатерина"


@dataclass
class ApiResponse:
    ok: bool
    status: int
    data: dict | None
    error: str | None


class AAClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = ""

    def _prefixes(self):
        # пробуем несколько вариантов (у тебя на деплоях бывает /api)
        return ["", "/api"]

    def _mk(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "OzonatorAgentsClient/1.0",
        }
        if AA_BEARER:
            h["Authorization"] = f"Bearer {AA_BEARER}"
        if AA_ADMIN_TOKEN:
            h["X-Admin-Token"] = AA_ADMIN_TOKEN
        return h

    def _request_raw(self, method: str, url: str, body: bytes | None, headers: dict, timeout: int = HTTP_TIMEOUT) -> ApiResponse:
        req = urllib_request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                txt = raw.decode("utf-8") if raw else ""
                return ApiResponse(True, r.status, json.loads(txt) if txt else {}, None)
        except urllib_error.HTTPError as e:
            status = int(getattr(e, "code", 0) or 0)
            try:
                raw = e.read()
                txt = raw.decode("utf-8") if raw else ""
                payload = json.loads(txt) if txt else {}
            except Exception:
                payload = None
            return ApiResponse(False, status, payload, f"HTTPError {status}")
        except urllib_error.URLError as e:
            return ApiResponse(False, 0, None, f"URLError: {e}")
        except Exception as e:
            return ApiResponse(False, 0, None, f"{e.__class__.__name__}: {e}")

    def _request_json(self, method: str, url: str, body: dict | None, timeout: int = HTTP_TIMEOUT) -> ApiResponse:
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib_request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                txt = raw.decode("utf-8") if raw else ""
                return ApiResponse(True, r.status, json.loads(txt) if txt else {}, None)
        except urllib_error.HTTPError as e:
            status = int(getattr(e, "code", 0) or 0)
            try:
                raw = e.read()
                txt = raw.decode("utf-8") if raw else ""
                payload = json.loads(txt) if txt else {}
            except Exception:
                payload = None
            return ApiResponse(False, status, payload, f"HTTPError {status}")
        except urllib_error.URLError as e:
            return ApiResponse(False, 0, None, f"URLError: {e}")
        except Exception as e:
            return ApiResponse(False, 0, None, f"{e.__class__.__name__}: {e}")

    def create_task(self, payload: dict) -> int:
        body = {
            "target_agent": "AA",
            "task_type": "user_task",
            "priority": 100,
            "payload": payload,
        }

        last_err = None
        for _ in range(CREATE_RETRIES + 1):
            for pref in self._prefixes():
                self.api_prefix = pref
                url = self._mk("/tasks/create")
                resp = self._request_json("POST", url, body)
                if resp.ok and resp.data and resp.data.get("task") and resp.data["task"].get("id"):
                    return int(resp.data["task"]["id"])
                last_err = resp.error or "create_task_failed"
        raise RuntimeError(last_err or "create_task_failed")

    def run_task(self, task_id: int) -> None:
        """"Пинок" оркестрации.

        Важно: на Render возможен таймаут на cold start — это НЕ ошибка, polling продолжится.
        Но если сервер явно отвечает 4xx (например, неверный путь/токен) — показываем ошибку,
        чтобы не зависать бесконечно.
        """
        had_timeout = False
        last_4xx: ApiResponse | None = None

        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/aa/run-task/{task_id}")
            resp = self._request_json("POST", url, {})
            if resp.ok:
                return

            if resp.status == 0:
                had_timeout = True
                continue

            if 400 <= resp.status < 500:
                last_4xx = resp

        # Если были только таймауты/сетевые ошибки — считаем, что "пинок" мог сработать.
        if had_timeout and last_4xx is None:
            return

        if last_4xx is not None:
            raise RuntimeError(last_4xx.error or f"run_task_failed ({last_4xx.status})")


    def get_task(self, task_id: int) -> dict | None:
        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/tasks/{task_id}")
            resp = self._request_json("GET", url, None)
            if resp.ok and resp.data and resp.data.get("task"):
                return resp.data["task"]
        return None

    def get_logs(self, task_id: int) -> list[dict]:
        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/tasks/{task_id}/logs")
            resp = self._request_json("GET", url, None)
            if resp.ok and resp.data and isinstance(resp.data.get("logs"), list):
                return resp.data["logs"]
        return []

    def get_recent_history(self, user_key: str, user_name: str | None = None, limit: int = 30) -> list[dict]:
        """Пытается восстановить историю с сервера (если локальный файл отсутствует).

        Возвращает список элементов (в обратном хронологическом порядке на сервере),
        поэтому клиент дальше разворачивает в нормальный порядок.
        """
        limit = max(1, min(int(limit or 30), 200))
        q = {
            "limit": str(limit),
        }
        if user_key:
            q["user_key"] = user_key
        if user_name:
            q["user_name"] = str(user_name)

        query = urllib_parse.urlencode(q)

        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/history/recent?{query}")
            resp = self._request_json("GET", url, None)
            if resp.ok and resp.data and isinstance(resp.data.get("items"), list):
                return resp.data.get("items") or []
        return []


    def upload_file(self, task_id: int, file_path: str) -> None:

        upload_name, ctype, content, note = _prepare_file_for_upload(file_path)

        boundary = "----ozonatorboundary" + uuid.uuid4().hex
        head = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{upload_name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + content + tail

        # Динамический таймаут: чем больше тело — тем больше даём времени.
        # Для превью обычно это <1MB и таймаут становится коротким.
        dyn = int(len(body) / 180_000) + 45  # ~0.18MB/s + запас
        upload_timeout = max(UPLOAD_TIMEOUT, dyn)
        upload_timeout = min(upload_timeout, 900)

        last_err = None
        for attempt in range(UPLOAD_RETRIES + 1):
            for pref in self._prefixes():
                self.api_prefix = pref
                url = self._mk(f"/tasks/{task_id}/files/upload")
                headers = self._headers().copy()
                headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
                headers["Accept"] = "application/json"

                resp = self._request_raw("POST", url, body, headers, timeout=upload_timeout)
                if resp.ok:
                    return

                detail = ""
                if isinstance(resp.data, dict):
                    detail = str(resp.data.get("detail") or resp.data.get("message") or "").strip()
                last_err = (f"{resp.error or 'upload_failed'} (status={resp.status}) {detail}").strip()

            # сетевые/таймауты — ретраим с небольшим backoff
            time.sleep(0.9 * (attempt + 1))

        # если дошли сюда — всё плохо
        raise RuntimeError((last_err or "upload_failed") + f"; note={note}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Ozonator Agents Client")
        self.geometry("940x680")
        self.minsize(840, 600)

        self.client = AAClient(DEFAULT_AA_BASE_URL)

        self.current_task_id: int | None = None
        self._polling = False
        self._poll_fast_until = 0.0
        self._poll_inflight = False

        self._typing_range = None
        self._conversation_history: list[dict[str, str]] = []
        self._transcript: list[dict[str, str]] = []
        self._state_path = os.path.join(_app_data_dir(), STATE_FILE_NAME)
        self._config_path = os.path.join(_app_data_dir(), CONFIG_FILE_NAME)
        self.user_key = _load_or_create_user_key(self._config_path, AA_BEARER, AA_ADMIN_TOKEN)
        self._attached_files: list[str] = []
        self._last_task_status: str = ""
        self._poll_started_at: float = 0.0

        # send/create_task tracking (чтобы не зависать без task_id)
        self._sending: bool = False
        self._sending_started_at: float | None = None
        self._send_stage: str = ""
        self._last_send_error: str | None = None
        self._send_nonce: str = uuid.uuid4().hex

        self._build_ui()
        self._bind_hotkeys()

        self._restore_on_startup()

        self.after(FAST_POLL_MS, self._tick)

    def _bind_hotkeys(self):
        self.bind_all("<Control-l>", lambda _e: self._clear_input())
        self.bind_all("<Control-L>", lambda _e: self._clear_input())
        self.bind_all("<Return>", lambda _e: self._on_send())

    def _build_ui(self):
        # Header
        header = tk.Frame(self, padx=10, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        # Avatar (simple circle)
        self.avatar = tk.Canvas(header, width=44, height=44, highlightthickness=0)
        self.avatar.pack(side=tk.LEFT)
        self._draw_avatar()

        name_frame = tk.Frame(header)
        name_frame.pack(side=tk.LEFT, padx=10)

        self.lbl_name = tk.Label(name_frame, text=AA_DISPLAY_NAME, font=("Segoe UI", 14, "bold"))
        self.lbl_name.pack(anchor="w")
        self.lbl_status = tk.Label(name_frame, text="● online", font=("Segoe UI", 10))
        self.lbl_status.pack(anchor="w")

        self.btn_logs = tk.Button(header, text="Логи", command=self._open_logs)
        self.btn_logs.pack(side=tk.RIGHT)

        # Chat area
        body = tk.Frame(self, padx=10, pady=6)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.chat = tk.Text(body, wrap="word", state="disabled", font=("Segoe UI", 11))
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(body, command=self.chat.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.configure(yscrollcommand=scroll.set)

        self.chat.tag_configure("ts", foreground="#666666")
        self.chat.tag_configure("user", font=("Segoe UI", 11, "bold"))
        self.chat.tag_configure("aa", font=("Segoe UI", 11, "bold"))

        # Input row
        bottom = tk.Frame(self, padx=10, pady=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_files = tk.Button(bottom, text="Файлы", command=self._pick_files)
        self.btn_files.pack(side=tk.LEFT)

        self.entry = tk.Entry(bottom, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.entry.focus_set()

        self.btn_send = tk.Canvas(bottom, width=46, height=46, highlightthickness=0)
        self.btn_send.pack(side=tk.RIGHT, padx=(10, 0))
        self._draw_send_button()
        self.btn_send.bind("<Button-1>", lambda _e: self._on_send())

    def _draw_avatar(self):
        self.avatar.delete("all")
        self.avatar.create_oval(2, 2, 42, 42, fill="#2b2b2b", outline="")
        self.avatar.create_text(22, 22, text="Е", fill="white", font=("Segoe UI", 16, "bold"))

    def _draw_send_button(self):
        self.btn_send.delete("all")
        self.btn_send.create_oval(2, 2, 44, 44, fill="#2b2b2b", outline="")
        # small arrow
        self.btn_send.create_polygon(20, 15, 32, 23, 20, 31, 22, 23, fill="white", outline="white")

    def _append_with_stamp(self, stamp: str, who: str, text: str):
        self.chat.configure(state="normal")
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert(tk.END, "\n")
        self.chat.insert(tk.END, f"{stamp}\n", ("ts",))
        tag = "user" if who == "Ты" else "aa"
        self.chat.insert(tk.END, f"{who}\n", (tag,))
        self.chat.insert(tk.END, f"{text.strip()}\n")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

    def _append(self, who: str, text: str):
        self._append_with_stamp(datetime.now().strftime("%H:%M"), who, text)

    def _add_message(
        self,
        role: str,
        who: str,
        text: str,
        include_in_context: bool = True,
        stamp: str | None = None,
    ):
        stamp = stamp or datetime.now().strftime("%H:%M")
        self._append_with_stamp(stamp, who, text)

        # В LLM-контекст кладём только user/assistant.
        if include_in_context and role in {"user", "assistant"}:
            self._push_history(role, text)

        # Полный лог для восстановления UI
        self._transcript.append({
            "stamp": stamp,
            "role": role,
            "who": who,
            "content": text,
        })
        if len(self._transcript) > STATE_MAX_ITEMS:
            self._transcript = self._transcript[-STATE_MAX_ITEMS:]
        self._save_state()

    def _save_state(self):
        data = {
            "version": STATE_VERSION,
            "user_key": self.user_key,
            "updated_at": datetime.utcnow().isoformat(),
            "transcript": self._transcript[-STATE_MAX_ITEMS:],
        }
        _write_json_atomic(self._state_path, data)

    def _load_state(self) -> dict | None:
        st = _read_json_file(self._state_path)
        if not isinstance(st, dict):
            return None
        if int(st.get("version") or 0) != STATE_VERSION:
            return None
        tr = st.get("transcript")
        if not isinstance(tr, list):
            return None
        return st

    def _restore_on_startup(self):
        # 1) локальная история
        st = self._load_state()
        if st:
            tr = st.get("transcript") or []
            if isinstance(tr, list) and tr:
                self._transcript = []
                self._conversation_history = []
                for item in tr:
                    if not isinstance(item, dict):
                        continue
                    stamp = str(item.get("stamp") or "").strip() or "--:--"
                    who = str(item.get("who") or "").strip() or AA_DISPLAY_NAME
                    role = str(item.get("role") or "").strip() or "meta"
                    content = str(item.get("content") or "").strip()
                    if not content:
                        continue
                    self._append_with_stamp(stamp, who, content)
                    self._transcript.append({"stamp": stamp, "role": role, "who": who, "content": content})
                    if role in {"user", "assistant"}:
                        self._conversation_history.append({"role": role, "content": content})

        # 2) если локально пусто — пытаемся подтянуть с сервера
        if not self._transcript:
            self._restore_from_server_async()

    def _stamp_from_iso(self, iso_ts: str | None) -> str:
        if not iso_ts:
            return "--:--"
        try:
            # ожидаем формат вида 2026-03-04T12:34:56+00:00
            dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
            return dt.astimezone().strftime("%H:%M")
        except Exception:
            return "--:--"

    def _restore_from_server_async(self):
        def worker():
            try:
                items = self.client.get_recent_history(self.user_key, user_name="Александр", limit=24)
            except Exception:
                items = []

            if not items:
                return

            # сервер отдаёт DESC — разворачиваем в нормальный порядок
            items = list(reversed(items))

            def ui_apply():
                if self._transcript:
                    return
                self._apply_server_history(items)

            self.after(0, ui_apply)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_server_history(self, items: list[dict]):
        if not items:
            return

        self._transcript = []
        self._conversation_history = []

        for it in items:
            if not isinstance(it, dict):
                continue
            stamp = self._stamp_from_iso(it.get("created_at"))
            user_text = str(it.get("user_request") or "").strip()
            aa_text = str(it.get("final_answer") or "").strip()

            if user_text:
                self._append_with_stamp(stamp, "Ты", user_text)
                self._transcript.append({"stamp": stamp, "role": "user", "who": "Ты", "content": user_text})
                self._conversation_history.append({"role": "user", "content": user_text})

            if aa_text:
                self._append_with_stamp(stamp, AA_DISPLAY_NAME, aa_text)
                self._transcript.append({"stamp": stamp, "role": "assistant", "who": AA_DISPLAY_NAME, "content": aa_text})
                self._conversation_history.append({"role": "assistant", "content": aa_text})

        if len(self._conversation_history) > HISTORY_MAX_ITEMS:
            self._conversation_history = self._conversation_history[-HISTORY_MAX_ITEMS:]
        if len(self._transcript) > STATE_MAX_ITEMS:
            self._transcript = self._transcript[-STATE_MAX_ITEMS:]
        self._save_state()

    def _show_typing(self):
        # мгновенный отклик
        try:
            self._clear_typing_if_any()
            stamp = datetime.now().strftime("%H:%M")
            self.chat.configure(state="normal")
            if self.chat.index("end-1c") != "1.0":
                self.chat.insert(tk.END, "\n")
            start = self.chat.index(tk.END)
            self.chat.insert(tk.END, f"{stamp}\n", ("ts",))
            self.chat.insert(tk.END, f"{AA_DISPLAY_NAME}\n", ("aa",))
            self.chat.insert(tk.END, "…\n")
            end = self.chat.index(tk.END)
            self.chat.see(tk.END)
            self.chat.configure(state="disabled")
            self._typing_range = (start, end)
        except Exception:
            self._typing_range = None

    def _clear_typing_if_any(self):
        rng = getattr(self, "_typing_range", None)
        if not rng:
            return
        try:
            start, end = rng
            self.chat.configure(state="normal")
            self.chat.delete(start, end)
            self.chat.configure(state="disabled")
        except Exception:
            pass
        self._typing_range = None

    def _clear_input(self):
        self.entry.delete(0, tk.END)

    def _pick_files(self):
        paths = filedialog.askopenfilenames(title="Выбери файлы")
        if not paths:
            return
        self._attached_files = [str(p) for p in paths if p]
        self._update_status()

    def _clear_files(self):
        self._attached_files = []
        self._update_status()

    def _update_status(self):
        parts = ["● online"]
        if self._attached_files:
            parts.append(f"файлов: {len(self._attached_files)}")
        if self.current_task_id is not None:
            parts.append(f"задача #{self.current_task_id}")
        if self._last_task_status:
            parts.append(f"{self._last_task_status}")
        if self._sending and self._send_stage:
            parts.append(f"отправка: {self._send_stage}")
        self.lbl_status.config(text=" · ".join(parts))


    def _build_task_payload(self, user_text: str) -> dict:
        # user_prefs (как ты попросил)
        user_prefs = {
            "user_name": "Александр",
            "pronoun": "ты",
            "addressing_default": "Саша",
            "addressing_variants": ["Александр", "Саша", "Сашечка", "Александр Николаевич"],
            "never_discuss_ai": True,
        }

        # history
        history = list(self._conversation_history)[-HISTORY_MAX_ITEMS:]

        attachments_meta = []
        for p in self._attached_files:
            try:
                attachments_meta.append({
                    "name": os.path.basename(p) or "file",
                    "size_bytes": int(os.path.getsize(p)),
                })
            except Exception:
                attachments_meta.append({"name": os.path.basename(p) or "file"})

        return {
            "user_request": user_text,
            "llm_provider": DEFAULT_LLM_PROVIDER,
            "llm_model": DEFAULT_LLM_MODEL,
            "client_meta": {"source": "desktop", "client": "OzonatorAgentsClient"},
            "user_key": self.user_key,
            "user_prefs": user_prefs,
            "conversation_history": history,
            "attachments": attachments_meta,
        }

    def _push_history(self, role: str, content: str):
        self._conversation_history.append({"role": role, "content": content})
        if len(self._conversation_history) > HISTORY_MAX_ITEMS:
            self._conversation_history = self._conversation_history[-HISTORY_MAX_ITEMS:]

    def _on_send(self):
        user_text = self.entry.get().strip()
        if not user_text:
            return

        self._clear_input()
        self._add_message("user", "Ты", user_text)
        if self._attached_files:
            names = [os.path.basename(p) or "file" for p in self._attached_files]
            self._add_message("meta", "Ты", "Прикреплено файлов: " + ", ".join(names), include_in_context=False)

        self._show_typing()

        payload = self._build_task_payload(user_text)

        # reset per-send state
        self._polling = False
        self._poll_inflight = False
        self._last_task_status = ""
        self.current_task_id = None

        # start send tracking
        self._sending = True
        self._sending_started_at = time.time()
        self._send_stage = "create_task"
        self._last_send_error = None
        self._send_nonce = uuid.uuid4().hex
        local_nonce = self._send_nonce
        self._update_status()

        def worker(nonce: str):
            try:
                task_id = self.client.create_task(payload)

                def ui_set_task_id():
                    if nonce != self._send_nonce:
                        return
                    self.current_task_id = task_id
                    self._send_stage = "upload_files"
                    self._update_status()

                self.after(0, ui_set_task_id)

                # upload attachments before orchestration
                for p in list(self._attached_files):
                    self.client.upload_file(task_id, p)

                self.after(0, self._clear_files)

                def ui_set_run_stage():
                    if nonce != self._send_nonce:
                        return
                    self._send_stage = "run_task"
                    self._update_status()

                self.after(0, ui_set_run_stage)

                self.client.run_task(task_id)

                def ui_start_polling():
                    if nonce != self._send_nonce:
                        return
                    self._sending = False
                    self._send_stage = ""
                    self._poll_started_at = time.time()
                    self._polling = True
                    self._poll_fast_until = time.time() + FAST_POLL_WINDOW_SEC
                    self._update_status()

                self.after(0, ui_start_polling)

            except Exception as e:
                err = f"{e}"

                def ui_fail():
                    if nonce != self._send_nonce:
                        return
                    stage = self._send_stage or "-"
                    self._sending = False
                    self._send_stage = ""
                    self._last_send_error = err
                    self._polling = False
                    self.current_task_id = None
                    self._update_status()
                    self._on_error(f"Не удалось отправить задачу ({stage}): {err}")

                self.after(0, ui_fail)

        threading.Thread(target=worker, args=(local_nonce,), daemon=True).start()

    def _on_error(self, msg: str):
        self._clear_typing_if_any()
        messagebox.showerror("Ошибка", msg)

    def _current_poll_interval_ms(self) -> int:
        return FAST_POLL_MS if time.time() < self._poll_fast_until else NORMAL_POLL_MS

    def _tick(self):
        interval = self._current_poll_interval_ms()

        # Если task_id не приходит (create_task завис/сеть/таймаут) — не зависаем в "…".
        if self._sending and self.current_task_id is None and self._sending_started_at:
            if (time.time() - float(self._sending_started_at)) > SEND_TIMEOUT_SEC:
                # инвалидируем поздние ответы фонового потока
                self._send_nonce = uuid.uuid4().hex
                self._sending = False
                self._send_stage = ""
                self._last_send_error = f"timeout_wait_task_id_{SEND_TIMEOUT_SEC}s"
                self._clear_typing_if_any()
                msg = (
                    f"Не удалось получить task_id за {SEND_TIMEOUT_SEC} сек. "
                    f"Проверь доступность сервера ({self.client.base_url}) и сеть. Нажми «Логи» — покажу диагностику отправки."
                )
                self._add_message("assistant", AA_DISPLAY_NAME, msg)
                self._update_status()

        if self._polling and self.current_task_id is not None and not self._poll_inflight:
            self._poll_inflight = True

            def poll_worker(task_id: int):
                try:
                    task = self.client.get_task(task_id)
                    self.after(0, lambda: self._handle_task_update(task_id, task))
                finally:
                    self._poll_inflight = False

            threading.Thread(target=poll_worker, args=(self.current_task_id,), daemon=True).start()

        self.after(interval, self._tick)

    def _handle_task_update(self, task_id: int, task: dict | None):
        if not task:
            return

        status = str(task.get("status") or "").upper()

        if status and status != self._last_task_status:
            self._last_task_status = status
            self._update_status()
        result = task.get("result") if isinstance(task.get("result"), dict) else {}

        # финальный ответ всегда из task.result.final_answer
        final_answer = ""
        if isinstance(result, dict):
            final_answer = str(result.get("final_answer") or "").strip()

        if final_answer:
            self._polling = False
            self._clear_typing_if_any()
            self._add_message("assistant", AA_DISPLAY_NAME, final_answer)
            return

        # Если статус финальный, но финальный ответ пустой — выводим понятное сообщение и останавливаемся.
        if status in {"DONE", "REVIEW_NEEDS_ATTENTION"}:
            self._polling = False
            self._clear_typing_if_any()
            msg = "Задача завершилась, но финальный ответ пуст. Открой «Логи» и пришли скрин последних строк."
            self._add_message("assistant", AA_DISPLAY_NAME, msg)
            return

        # Защита от бесконечного ожидания: если больше 90 секунд нет результата — подскажем открыть логи.
        if self._poll_started_at and (time.time() - self._poll_started_at) > 90:
            self._polling = False
            self._clear_typing_if_any()
            msg = "Я не получила финальный ответ за 90 секунд. Нажми «Логи» и пришли скрин последней части — разберу, где застряло."
            self._add_message("assistant", AA_DISPLAY_NAME, msg)
            return

        if status in {"FAILED"}:
            self._polling = False
            self._clear_typing_if_any()
            err = str(task.get("error_message") or "Задача завершилась с ошибкой").strip()
            self._add_message("assistant", AA_DISPLAY_NAME, err)

    def _open_logs(self):
        # Если task_id ещё нет — показываем диагностику отправки (иначе пользователь не может понять, что сломалось).
        if self.current_task_id is None:
            win = tk.Toplevel(self)
            win.title("Логи — диагностика отправки")
            win.geometry("900x420")

            txt = tk.Text(win, wrap="word", font=("Consolas", 10))
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            sc = tk.Scrollbar(win, command=txt.yview)
            sc.pack(side=tk.RIGHT, fill=tk.Y)
            txt.configure(yscrollcommand=sc.set)

            elapsed = 0.0
            if self._sending_started_at:
                elapsed = max(0.0, time.time() - float(self._sending_started_at))

            txt.insert(tk.END, f"server: {self.client.base_url}\n")
            txt.insert(tk.END, f"HTTP_TIMEOUT: {HTTP_TIMEOUT}s\n")
            txt.insert(tk.END, f"UPLOAD_TIMEOUT: {UPLOAD_TIMEOUT}s\n")
            txt.insert(tk.END, f"UPLOAD_RETRIES: {UPLOAD_RETRIES}\n")
            txt.insert(tk.END, f"sending: {self._sending}\n")
            txt.insert(tk.END, f"stage: {self._send_stage or '-'}\n")
            txt.insert(tk.END, f"elapsed_sec: {elapsed:.1f}\n")
            if self._last_send_error:
                txt.insert(tk.END, f"last_error: {self._last_send_error}\n")

            txt.insert(tk.END, "\nПодсказка: если stage=create_task и elapsed_sec растёт — это сеть/DNS/таймаут или сервер недоступен.\n")
            txt.configure(state="disabled")
            return

        logs = self.client.get_logs(self.current_task_id)

        win = tk.Toplevel(self)
        win.title(f"Логи задачи #{self.current_task_id}")
        win.geometry("900x600")

        txt = tk.Text(win, wrap="word", font=("Consolas", 10))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sc = tk.Scrollbar(win, command=txt.yview)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        txt.configure(yscrollcommand=sc.set)

        for item in logs:
            ts = item.get("created_at") or ""
            actor = item.get("actor_agent") or ""
            ev = item.get("event_type") or ""
            lvl = item.get("level") or ""
            msg = item.get("message") or ""
            txt.insert(tk.END, f"{ts} [{lvl}] {actor} {ev} — {msg}\n")

        txt.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
