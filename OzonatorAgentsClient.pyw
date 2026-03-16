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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from tkinter import filedialog, messagebox
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


DEFAULT_AA_BASE_URL = (os.getenv("OZONATOR_AA_BASE_URL") or "https://ozonator-aa-dev.onrender.com").rstrip("/")
AA_BEARER = (os.getenv("OZONATOR_AA_BEARER") or "").strip()
AA_ADMIN_TOKEN = (os.getenv("OZONATOR_AA_ADMIN_TOKEN") or "").strip()

DEFAULT_LLM_PROVIDER = (os.getenv("OZONATOR_LLM_PROVIDER") or "groq").strip()
DEFAULT_LLM_MODEL = (os.getenv("OZONATOR_LLM_MODEL") or "llama-3.3-70b-versatile").strip()

CREATE_RETRIES = 2
HTTP_TIMEOUT = int(os.getenv("OZONATOR_HTTP_TIMEOUT") or 90)
UPLOAD_TIMEOUT = int(os.getenv("OZONATOR_UPLOAD_TIMEOUT") or max(180, HTTP_TIMEOUT))
UPLOAD_RETRIES = int(os.getenv("OZONATOR_UPLOAD_RETRIES") or 2)
SEND_TIMEOUT_SEC = 75
FAST_POLL_MS = 250
NORMAL_POLL_MS = 1000
FAST_POLL_WINDOW_SEC = 12

CLIENT_IMAGE_MAX_BYTES = int(os.getenv("OZONATOR_CLIENT_IMAGE_MAX_BYTES") or 900_000)
CLIENT_IMAGE_MAX_DIM = int(os.getenv("OZONATOR_CLIENT_IMAGE_MAX_DIM") or 1600)
CLIENT_IMAGE_PREVIEW_ONLY = (os.getenv("OZONATOR_CLIENT_IMAGE_PREVIEW_ONLY") or "1").strip().lower() not in ("0", "false", "no")

try:
    from PIL import Image, ImageGrab  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    ImageGrab = None  # type: ignore
    PIL_AVAILABLE = False


STATE_VERSION = 1
STATE_MAX_ITEMS = max(50, int(os.getenv("OZONATOR_STATE_MAX_ITEMS") or 200))
STATE_FILE_NAME = "chat_state_v1.json"
CONFIG_FILE_NAME = "client_config_v1.json"

GEO_STATE_FILE_NAME = "geo_state_v1.json"
GEO_TTL_SEC = max(300, int(os.getenv("OZONATOR_GEO_TTL_SEC") or str(6 * 60 * 60)))

HISTORY_MAX_ITEMS = int(os.getenv("OZONATOR_HISTORY_MAX_ITEMS") or 80)
HISTORY_MAX_CHARS = int(os.getenv("OZONATOR_HISTORY_MAX_CHARS") or 30000)
HISTORY_MAX_EACH = int(os.getenv("OZONATOR_HISTORY_MAX_EACH") or 1400)
HISTORY_HARD_MAX = int(os.getenv("OZONATOR_HISTORY_HARD_MAX") or 400)

AA_DISPLAY_NAME = "Екатерина"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _app_data_dir() -> str:
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
    cfg = _read_json_file(config_path) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["version"] = 1
    cfg["user_key"] = key
    _write_json_atomic(config_path, cfg)
    return key


def _is_image_name(file_name: str, content_type: str) -> bool:
    ct = (content_type or "").lower().strip()
    if ct.startswith("image/"):
        return True
    ext = (file_name or "").lower().rsplit(".", 1)
    ext = ext[-1] if len(ext) == 2 else ""
    return ext in {"jpg", "jpeg", "png", "webp", "gif"}


def _make_image_preview(raw: bytes, max_dim: int, max_bytes: int):
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
    file_name = os.path.basename(file_path) or "file"
    ctype = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

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
        return [self.api_prefix] if self.api_prefix else ["", "/api/v1"]

    def _full_url(self, path: str, prefix: str = "") -> str:
        return f"{self.base_url}{prefix}{path}"

    def _headers_json(self) -> dict:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if AA_BEARER:
            h["Authorization"] = f"Bearer {AA_BEARER}"
        if AA_ADMIN_TOKEN:
            h["X-Admin-Token"] = AA_ADMIN_TOKEN
        return h

    def _headers_plain(self) -> dict:
        h = {"Accept": "application/json"}
        if AA_BEARER:
            h["Authorization"] = f"Bearer {AA_BEARER}"
        if AA_ADMIN_TOKEN:
            h["X-Admin-Token"] = AA_ADMIN_TOKEN
        return h

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: int | None = None) -> ApiResponse:
        body = None
        headers = self._headers_json() if payload is not None else self._headers_plain()
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_err = None
        for prefix in self._prefixes():
            url = self._full_url(path, prefix)
            req = urllib_request.Request(url, method=method.upper(), data=body, headers=headers)
            try:
                with urllib_request.urlopen(req, timeout=timeout or HTTP_TIMEOUT) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw) if raw else {}
                    self.api_prefix = prefix
                    return ApiResponse(True, getattr(resp, "status", 200), data, None)
            except urllib_error.HTTPError as e:
                raw = ""
                try:
                    raw = e.read().decode("utf-8")
                except Exception:
                    pass
                try:
                    data = json.loads(raw) if raw else {}
                except Exception:
                    data = None
                if getattr(e, "code", None) == 404:
                    last_err = ApiResponse(False, e.code, data, raw or str(e))
                    continue
                return ApiResponse(False, e.code, data, raw or str(e))
            except Exception as e:
                last_err = ApiResponse(False, 0, None, str(e))
                continue
        return last_err or ApiResponse(False, 0, None, "unknown error")

    def create_task(self, payload: dict) -> int:
        last_error = None
        for attempt in range(CREATE_RETRIES + 1):
            r = self._request("POST", "/tasks/create", payload)
            if r.ok:
                task = (r.data or {}).get("task") or {}
                task_id = task.get("id")
                if not task_id:
                    raise RuntimeError("AA не вернул task.id")
                return int(task_id)
            last_error = r.error or f"HTTP {r.status}"
            if r.status and 400 <= r.status < 500:
                break
            if attempt < CREATE_RETRIES:
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Не удалось создать задачу: {last_error}")

    def upload_file(self, task_id: int, file_path: str) -> dict:
        url = self._full_url(f"{self.api_prefix}/tasks/{task_id}/files/upload")

        try:
            file_name, content_type, file_bytes, note = _prepare_file_for_upload(file_path)
        except Exception as e:
            raise RuntimeError(f"Не удалось подготовить файл к отправке: {e}")

        boundary = f"----OzonatorBoundary{uuid.uuid4().hex}"
        body = build_multipart_form_data(boundary, "file", file_name, content_type, file_bytes)
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        if AA_BEARER:
            headers["Authorization"] = f"Bearer {AA_BEARER}"
        if AA_ADMIN_TOKEN:
            headers["X-Admin-Token"] = AA_ADMIN_TOKEN

        last_err = None
        for attempt in range(UPLOAD_RETRIES + 1):
            req = urllib_request.Request(url, method="POST", data=body, headers=headers)
            try:
                with urllib_request.urlopen(req, timeout=UPLOAD_TIMEOUT) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw) if raw else {}
                    file_obj = (data or {}).get("file") or {}
                    if isinstance(file_obj, dict):
                        file_obj.setdefault("client_note", note)
                    return file_obj
            except urllib_error.HTTPError as e:
                raw = ""
                try:
                    raw = e.read().decode("utf-8")
                except Exception:
                    pass
                last_err = raw or str(e)
                if getattr(e, "code", 0) and 400 <= e.code < 500:
                    break
            except Exception as e:
                last_err = str(e)
            if attempt < UPLOAD_RETRIES:
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Не удалось загрузить файл: {last_err}")

    def run_task(self, task_id: int) -> dict:
        r = self._request("POST", f"/aa/run-task/{task_id}", {})
        if not r.ok:
            raise RuntimeError(f"Не удалось запустить задачу: {r.error or r.status}")
        return r.data or {}

    def get_task(self, task_id: int) -> dict | None:
        r = self._request("GET", f"/tasks/{task_id}")
        if not r.ok:
            return None
        return (r.data or {}).get("task")

    def get_logs(self, task_id: int) -> list[dict]:
        r = self._request("GET", f"/tasks/{task_id}/logs")
        if not r.ok:
            return []
        return list((r.data or {}).get("logs") or [])

    def list_task_files(self, task_id: int) -> list[dict]:
        r = self._request("GET", f"/tasks/{task_id}/files")
        if not r.ok:
            raise RuntimeError(r.error or f"HTTP {r.status}")
        return list((r.data or {}).get("files") or [])

    def download_task_file(self, task_id: int, file_id: int) -> bytes:
        url = self._full_url(f"{self.api_prefix}/tasks/{task_id}/files/{file_id}/download")
        headers = self._headers_plain()
        req = urllib_request.Request(url, method="GET", headers=headers)
        with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read()

    def history_recent(self, user_key: str | None, user_name: str | None, limit: int = 30) -> list[dict]:
        params = {}
        if user_key:
            params["user_key"] = user_key
        if user_name:
            params["user_name"] = user_name
        params["limit"] = str(limit)
        qs = urllib_parse.urlencode(params)
        path = "/history/recent"
        if qs:
            path += "?" + qs
        r = self._request("GET", path)
        if not r.ok:
            return []
        return list((r.data or {}).get("items") or [])


def build_multipart_form_data(boundary: str, field_name: str, file_name: str, content_type: str, file_bytes: bytes) -> bytes:
    b = io.BytesIO()
    sep = f"--{boundary}\r\n".encode("utf-8")
    b.write(sep)
    b.write(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'.encode("utf-8")
    )
    b.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    b.write(file_bytes)
    b.write(b"\r\n")
    b.write(f"--{boundary}--\r\n".encode("utf-8"))
    return b.getvalue()


def now_hhmm() -> str:
    return time.strftime("%H:%M")


def _safe_text(s) -> str:
    if s is None:
        return ""
    return str(s)


def _clip_text(s: str, max_len: int) -> str:
    s = _safe_text(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _sanitize_for_context(role: str, author: str, text: str) -> str:
    role = _safe_text(role).strip().lower()
    author = _safe_text(author).strip() or ("Ты" if role == "user" else AA_DISPLAY_NAME)
    text = _safe_text(text).replace("\r\n", "\n").strip()
    text = _clip_text(text, HISTORY_MAX_EACH)
    return f"{author}: {text}"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ozonator Agents Client")
        self.geometry("980x620")
        self.minsize(760, 520)

        self.client = AAClient(DEFAULT_AA_BASE_URL)

        self.current_task_id: int | None = None
        self._last_task_status: str = ""
        self._typing_item_id = None
        self._attached_files: list[str] = []
        self._message_items: list[dict] = []
        self._polling = False
        self._poll_inflight = False
        self._poll_started_at = 0.0
        self._poll_fast_until = 0.0
        self._sending = False
        self._sending_started_at = 0.0
        self._send_stage = ""
        self._last_send_error: str | None = None
        self._send_nonce = uuid.uuid4().hex
        self._chat_download_links: dict[str, dict] = {}
        self._announced_file_ids_by_task: dict[int, set[int]] = {}

        self._config_path = os.path.join(_app_data_dir(), CONFIG_FILE_NAME)
        self._state_path = os.path.join(_app_data_dir(), STATE_FILE_NAME)
        self._user_key = _load_or_create_user_key(self._config_path, AA_BEARER, AA_ADMIN_TOKEN)
        self._share_geo = _env_bool("OZONATOR_SHARE_GEO", True)
        self._share_geo_prefs = _read_json_file(self._config_path) or {}
        if isinstance(self._share_geo_prefs, dict) and "share_geo" in self._share_geo_prefs:
            try:
                self._share_geo = bool(self._share_geo_prefs.get("share_geo"))
            except Exception:
                pass

        self._build_ui()
        self._install_context_menus()
        self._restore_history()
        self._tick()

    def _geo_state_path(self) -> str:
        return os.path.join(_app_data_dir(), GEO_STATE_FILE_NAME)

    def _load_geo_cache(self) -> dict | None:
        data = _read_json_file(self._geo_state_path())
        if not isinstance(data, dict):
            return None
        try:
            fetched_at = data.get("fetched_at")
            if fetched_at:
                dt = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
                if age > GEO_TTL_SEC:
                    return None
        except Exception:
            return None
        geo = data.get("geo")
        return geo if isinstance(geo, dict) else None

    def _fetch_geo_by_ip(self) -> dict | None:
        candidates = [
            "https://ipapi.co/json/",
            "https://ipwho.is/",
        ]
        headers = {"User-Agent": "ozonator-agents-client/1.0", "Accept": "application/json"}

        for url in candidates:
            try:
                req = urllib_request.Request(url, headers=headers, method="GET")
                with urllib_request.urlopen(req, timeout=8) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                    data = json.loads(raw)

                geo = {
                    "ip": data.get("ip"),
                    "lat": data.get("latitude") if data.get("latitude") is not None else data.get("latitude_decimal"),
                    "lon": data.get("longitude") if data.get("longitude") is not None else data.get("longitude_decimal"),
                    "city": data.get("city"),
                    "region": data.get("region") or data.get("region_name"),
                    "country": data.get("country_name") or data.get("country"),
                    "source": "ip",
                    "timezone": data.get("timezone") if isinstance(data.get("timezone"), str) else (data.get("timezone", {}) or {}).get("id"),
                }

                success = data.get("success")
                if success is False:
                    continue

                lat = geo.get("lat")
                lon = geo.get("lon")
                if lat is None or lon is None:
                    continue
                try:
                    geo["lat"] = float(lat)
                    geo["lon"] = float(lon)
                except Exception:
                    continue
                return geo
            except Exception:
                continue
        return None

    def _get_geo(self) -> dict | None:
        if not self._share_geo:
            return None
        cached = self._load_geo_cache()
        if cached:
            return cached
        geo = self._fetch_geo_by_ip()
        if geo:
            _write_json_atomic(
                self._geo_state_path(),
                {
                    "version": 1,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "geo": geo,
                },
            )
        return geo

    def _on_toggle_geo(self):
        self._share_geo = bool(self.var_share_geo.get())
        cfg = _read_json_file(self._config_path) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["version"] = 1
        cfg["user_key"] = self._user_key
        cfg["share_geo"] = self._share_geo
        _write_json_atomic(self._config_path, cfg)
        try:
            if getattr(self, "_geo_status_var", None) is not None:
                self._geo_status_var.set(self._format_geo_status())
        except Exception:
            pass

    def _build_ui(self):
        top = tk.Frame(self, padx=10, pady=10)
        top.pack(fill=tk.X)

        left = tk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(left, text="Екатерина", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        self.status_var = tk.StringVar(value="online")
        self.status_label = tk.Label(left, textvariable=self.status_var, font=("Segoe UI", 10))
        self.status_label.pack(anchor="w", pady=(2, 0))

        right = tk.Frame(top)
        right.pack(side=tk.RIGHT)

        tk.Button(right, text="Настройки", command=self._open_settings).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(right, text="Скачать", command=self._download_or_open_task_files).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(right, text="Логи", command=self._open_logs).pack(side=tk.LEFT)

        mid = tk.Frame(self, padx=10)
        mid.pack(fill=tk.BOTH, expand=True)

        self.chat = tk.Text(mid, wrap="word", state="disabled", font=("Segoe UI", 11))
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(mid, command=self.chat.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.configure(yscrollcommand=scroll.set)

        bottom = tk.Frame(self, padx=10, pady=10)
        bottom.pack(fill=tk.X)

        self.files_btn = tk.Button(bottom, text="Файлы", command=self._pick_files)
        self.files_btn.pack(side=tk.LEFT)

        self.entry = tk.Text(bottom, height=4, wrap="word", font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.entry.bind("<Return>", self._on_enter_pressed)
        self.entry.bind("<Shift-Return>", self._on_shift_enter_pressed)
        self.entry.bind("<Control-v>", self._on_paste)
        self.entry.bind("<Control-V>", self._on_paste)

        send_wrap = tk.Frame(bottom, width=56, height=56)
        send_wrap.pack(side=tk.RIGHT)
        send_wrap.pack_propagate(False)

        self.send_btn = tk.Button(send_wrap, text="➤", font=("Segoe UI", 18), command=self._on_send)
        self.send_btn.pack(fill=tk.BOTH, expand=True)

        self.var_share_geo = tk.BooleanVar(value=self._share_geo)
        self._update_status()

    def _update_status(self):
        parts = ["online"]

        if self.current_task_id is not None:
            task_part = f"задача #{self.current_task_id}"
            if self._last_task_status:
                task_part += f" · {self._last_task_status}"
            parts.append(task_part)
        elif self._sending:
            stage_map = {
                "create_task": "создание задачи",
                "upload_files": "загрузка файлов",
                "run_task": "запуск задачи",
            }
            parts.append(stage_map.get(self._send_stage, self._send_stage or "отправка"))

        if self._attached_files:
            parts.append(f"файлов: {len(self._attached_files)}")

        self.status_var.set(" • ".join(parts))

    def _insert_chat(self, text: str, tag: str | None = None):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag if tag else ())
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _register_download_link(self, task_id: int, file_meta: dict) -> str:
        file_id = int(file_meta.get("id") or 0)
        tag = f"dl_{task_id}_{file_id}_{uuid.uuid4().hex[:8]}"
        self._chat_download_links[tag] = {"task_id": task_id, "file": dict(file_meta or {})}

        self.chat.tag_configure(tag, foreground="#0a58ca", underline=True)
        self.chat.tag_bind(tag, "<Enter>", lambda _e: self.chat.configure(cursor="hand2"))
        self.chat.tag_bind(tag, "<Leave>", lambda _e: self.chat.configure(cursor="xterm"))
        self.chat.tag_bind(tag, "<Button-1>", lambda _e, name=tag: self._download_from_chat_tag(name))
        return tag

    def _download_from_chat_tag(self, tag: str):
        meta = self._chat_download_links.get(tag) or {}
        task_id = int(meta.get("task_id") or 0)
        file_meta = meta.get("file") if isinstance(meta.get("file"), dict) else None
        if not task_id or not file_meta:
            messagebox.showerror("Скачать", "Не удалось определить файл для скачивания.")
            return
        self._save_task_file(task_id, file_meta)

    def _normalize_file_meta(self, item: dict | None) -> dict:
        meta = item if isinstance(item, dict) else {}
        return {
            "id": int(meta.get("id") or 0),
            "file_name": str(meta.get("file_name") or ""),
            "content_type": str(meta.get("content_type") or "application/octet-stream"),
            "size_bytes": int(meta.get("size_bytes") or 0),
            "created_at": meta.get("created_at"),
        }

    def _downloadable_files_from_task(self, task_id: int, task: dict | None = None) -> list[dict]:
        task = task if isinstance(task, dict) else None
        if task is None:
            try:
                task = self.client.get_task(task_id)
            except Exception:
                task = None

        result = task.get("result") if isinstance(task, dict) and isinstance(task.get("result"), dict) else {}
        result = result if isinstance(result, dict) else {}

        download_files = result.get("download_files") if isinstance(result.get("download_files"), list) else []
        normalized_downloads: list[dict] = []
        seen_download_ids: set[int] = set()
        for item in download_files:
            if not isinstance(item, dict):
                continue
            meta = self._normalize_file_meta(item)
            file_id = int(meta.get("id") or 0)
            if not file_id or file_id in seen_download_ids:
                continue
            seen_download_ids.add(file_id)
            normalized_downloads.append(meta)
        if normalized_downloads:
            return normalized_downloads

        try:
            files = self.client.list_task_files(task_id)
        except Exception:
            return []

        upload_ids_raw = result.get("user_upload_file_ids") if isinstance(result.get("user_upload_file_ids"), list) else []
        upload_ids = {int(x) for x in upload_ids_raw if str(x).isdigit()}

        upload_files = result.get("user_upload_files") if isinstance(result.get("user_upload_files"), list) else []
        upload_names = {
            str(item.get("file_name") or "").strip().lower()
            for item in upload_files
            if isinstance(item, dict) and str(item.get("file_name") or "").strip()
        }

        filtered: list[dict] = []
        seen_ids: set[int] = set()
        for item in files:
            if not isinstance(item, dict):
                continue
            meta = self._normalize_file_meta(item)
            file_id = int(meta.get("id") or 0)
            file_name = str(meta.get("file_name") or "").strip().lower()
            if not file_id or file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            if file_id in upload_ids:
                continue
            if file_name and file_name in upload_names:
                continue
            filtered.append(meta)
        return filtered

    def _save_task_file(self, task_id: int, meta: dict, parent=None) -> str | None:
        file_id = int(meta.get("id") or 0)
        file_name = str(meta.get("file_name") or f"file_{file_id}")

        save_path = filedialog.asksaveasfilename(
            initialfile=file_name,
            title="Куда сохранить файл",
            parent=parent or self,
        )
        if not save_path:
            return None

        try:
            content = self.client.download_task_file(task_id, file_id)
            with open(save_path, "wb") as f:
                f.write(content or b"")
        except Exception as e:
            messagebox.showerror("Скачать", f"Не удалось скачать файл: {e}", parent=parent or self)
            return None

        messagebox.showinfo("Скачать", f"Файл сохранён:\n{save_path}", parent=parent or self)
        return save_path

    def _announce_task_files_in_chat(self, task_id: int, files: list[dict] | None = None, task: dict | None = None):
        try:
            if files is None:
                files = self._downloadable_files_from_task(task_id, task=task)
        except Exception:
            return

        if not files:
            return

        announced = self._announced_file_ids_by_task.setdefault(int(task_id), set())
        new_items: list[dict] = []
        for item in files:
            try:
                file_id = int(item.get("id") or 0)
            except Exception:
                file_id = 0
            if not file_id or file_id in announced:
                continue
            announced.add(file_id)
            new_items.append(item)

        if not new_items:
            return

        self._insert_chat("Результаты для скачивания:\n", "files_hdr")
        for item in new_items:
            file_id = int(item.get("id") or 0)
            file_name = str(item.get("file_name") or f"file_{file_id}")
            size_bytes = int(item.get("size_bytes") or 0)
            tag = self._register_download_link(task_id, item)
            label = f"↓ {file_name}"
            if size_bytes > 0:
                label += f"  ({size_bytes} bytes)"
            self._insert_chat(label + "\n", tag)
        self._insert_chat("\n")

    def _download_or_open_task_files(self):
        if self.current_task_id is None:
            messagebox.showinfo("Скачать", "Сначала дождись задачи с результатами.")
            return

        try:
            task = self.client.get_task(self.current_task_id)
            files = self._downloadable_files_from_task(self.current_task_id, task=task)
        except Exception as e:
            messagebox.showerror("Скачать", f"Не удалось получить список файлов: {e}")
            return

        if not files:
            messagebox.showinfo("Скачать", "Для этой задачи результатов для скачивания пока нет.")
            return

        if len(files) == 1:
            self._save_task_file(self.current_task_id, files[0])
            return

        self._open_task_files(files=files)

    def _add_message(self, role: str, author: str, text: str, include_in_context: bool = True):
        ts = now_hhmm()
        header = f"\n{ts}\n{author}\n"
        self._insert_chat(header, "hdr")
        self._insert_chat(text + "\n", "msg")

        item = {
            "role": role,
            "author": author,
            "text": text,
            "ts": ts,
            "include_in_context": bool(include_in_context),
        }
        self._message_items.append(item)
        self._trim_message_items()
        self._persist_state()

    def _trim_message_items(self):
        if len(self._message_items) > HISTORY_HARD_MAX:
            self._message_items = self._message_items[-HISTORY_HARD_MAX:]

    def _show_typing(self):
        if self._typing_item_id is not None:
            return
        self.chat.configure(state="normal")
        self._typing_item_id = self.chat.index("end-1c")
        self.chat.insert("end", f"\n{AA_DISPLAY_NAME}\n…\n", "typing")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _clear_typing_if_any(self):
        if self._typing_item_id is None:
            return
        self.chat.configure(state="normal")
        try:
            self.chat.delete(self._typing_item_id, "end-1c")
        except Exception:
            pass
        self.chat.configure(state="disabled")
        self._typing_item_id = None

    def _pick_files(self):
        paths = filedialog.askopenfilenames(title="Выбери файлы")
        if not paths:
            return
        for p in paths:
            if p not in self._attached_files:
                self._attached_files.append(p)
        self._update_status()

    def _clear_files(self):
        self._attached_files.clear()
        self._update_status()

    def _clear_input(self):
        self.entry.delete("1.0", "end")

    def _on_enter_pressed(self, event):
        self._on_send()
        return "break"

    def _on_shift_enter_pressed(self, event):
        self.entry.insert("insert", "\n")
        return "break"

    def _state_payload(self) -> dict:
        items = self._message_items[-STATE_MAX_ITEMS:]
        return {
            "version": STATE_VERSION,
            "user_key": self._user_key,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "items": items,
        }

    def _persist_state(self):
        try:
            _write_json_atomic(self._state_path, self._state_payload())
        except Exception:
            pass

    def _restore_history(self):
        restored_any = False

        data = _read_json_file(self._state_path)
        if isinstance(data, dict) and data.get("version") == STATE_VERSION:
            items = data.get("items") or []
            if isinstance(items, list):
                for item in items[-STATE_MAX_ITEMS:]:
                    if not isinstance(item, dict):
                        continue
                    role = _safe_text(item.get("role")) or "assistant"
                    author = _safe_text(item.get("author")) or ("Ты" if role == "user" else AA_DISPLAY_NAME)
                    text = _safe_text(item.get("text"))
                    ts = _safe_text(item.get("ts")) or now_hhmm()
                    include_in_context = bool(item.get("include_in_context", True))
                    self.chat.configure(state="normal")
                    self.chat.insert("end", f"\n{ts}\n{author}\n", "hdr")
                    self.chat.insert("end", text + "\n", "msg")
                    self.chat.configure(state="disabled")
                    self._message_items.append(
                        {
                            "role": role,
                            "author": author,
                            "text": text,
                            "ts": ts,
                            "include_in_context": include_in_context,
                        }
                    )
                self.chat.see("end")
                restored_any = bool(items)

        if restored_any:
            return

        try:
            items = self.client.history_recent(self._user_key, None, limit=20)
        except Exception:
            items = []

        if not items:
            return

        for row in items:
            if not isinstance(row, dict):
                continue
            user_text = _safe_text(row.get("user_request") or row.get("content"))
            final_answer = _safe_text(row.get("final_answer"))
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            if not final_answer and isinstance(result, dict):
                final_answer = _safe_text(result.get("final_answer"))

            if user_text:
                self._add_message("user", "Ты", user_text)
            if final_answer:
                self._add_message("assistant", AA_DISPLAY_NAME, final_answer)

    def _context_messages(self) -> list[dict[str, str]]:
        items = [x for x in self._message_items if x.get("include_in_context", True)]
        items = items[-HISTORY_MAX_ITEMS:]

        chunks: list[dict[str, str]] = []
        total = 0
        for item in items:
            role = _safe_text(item.get("role")).strip().lower()
            if role not in {"user", "assistant"}:
                continue

            text = _safe_text(item.get("text")).replace("\r\n", "\n").strip()
            text = _clip_text(text, HISTORY_MAX_EACH)
            if not text:
                continue

            add_len = len(text) + 1
            if total + add_len > HISTORY_MAX_CHARS:
                break

            chunks.append({"role": role, "content": text})
            total += add_len
        return chunks

    def _build_task_payload(self, user_text: str) -> dict:
        prompt = user_text.strip()
        context = self._context_messages()

        payload = {
            "external_task_id": f"client-{uuid.uuid4().hex}",
            "task_type": "user_task",
            "source_agent": "USER",
            "target_agent": "AA",
            "priority": 100,
            "payload": {
                "geo": self._get_geo(),
                "user_key": self._user_key,
                "llm_model": DEFAULT_LLM_MODEL,
                "user_prefs": {
                    "pronoun": "ты",
                    "user_name": "Александр",
                    "assistant_name": AA_DISPLAY_NAME,
                },
                "user_name": "Александр",
                "prompt": prompt,
                "user_request": prompt,
                "brief": prompt,
                "conversation_history": context,
            },
        }
        return payload

    def _on_paste(self, event=None):
        try:
            txt = self.clipboard_get()
            if isinstance(txt, str) and txt != "":
                self.entry.insert("insert", txt)
                return "break"
        except Exception:
            pass

        try:
            clip = ImageGrab.grabclipboard() if ImageGrab is not None else None
            if Image is not None and isinstance(clip, Image.Image):  # type: ignore[attr-defined]
                base_dir = os.path.join(_app_data_dir(), "clipboard")
                os.makedirs(base_dir, exist_ok=True)
                fn = f"clipboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                path = os.path.join(base_dir, fn)
                clip.save(path, format="PNG")
                self._attached_files.append(path)
                self._add_message("meta", "Ты", f"Вставлено изображение из буфера: {fn}", include_in_context=False)
                self._update_status()
                return "break"
        except Exception:
            return None

        return None

    def _open_settings(self):
        try:
            if getattr(self, "_settings_win", None) is not None and self._settings_win.winfo_exists():
                self._settings_win.lift()
                self._settings_win.focus_force()
                return
        except Exception:
            pass

        win = tk.Toplevel(self)
        win.title("Настройки")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()
        self._settings_win = win

        frm = tk.Frame(win, padx=12, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Геопозиция", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        cb = tk.Checkbutton(
            frm,
            text="Передавать геопозицию (примерно по IP)",
            variable=self.var_share_geo,
            command=self._on_toggle_geo,
        )
        cb.pack(anchor="w", pady=(6, 4))

        self._geo_status_var = tk.StringVar(value=self._format_geo_status())
        tk.Label(frm, textvariable=self._geo_status_var, wraplength=380, justify="left").pack(anchor="w", pady=(0, 10))

        btns = tk.Frame(frm)
        btns.pack(fill=tk.X)

        tk.Button(btns, text="Обновить гео", command=self._refresh_geo_now).pack(side=tk.LEFT)
        tk.Button(btns, text="Очистить кэш", command=self._clear_geo_cache).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(btns, text="Закрыть", command=win.destroy).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", win.destroy)

    def _format_geo_status(self) -> str:
        try:
            geo = self._get_geo()
            if not geo:
                if getattr(self, "_share_geo", False):
                    return "Гео: включено, но ещё не определено. Нажми «Обновить гео»."
                return "Гео: выключено."

            parts = []
            for k in ("city", "region", "country"):
                v = geo.get(k)
                if v:
                    parts.append(str(v))
            place = ", ".join(parts) if parts else "неизвестно"
            lat = geo.get("lat")
            lon = geo.get("lon")
            if lat is not None and lon is not None:
                return f"Гео: {place} (lat={lat:.5f}, lon={lon:.5f})"
            return f"Гео: {place}"
        except Exception:
            return "Гео: не удалось определить."

    def _refresh_geo_now(self):
        try:
            if not getattr(self, "_share_geo", False):
                messagebox.showinfo("Геопозиция", "Сначала включи передачу геопозиции.")
                return
            geo = self._fetch_geo_by_ip()
            if geo:
                path = self._geo_state_path()
                _write_json_atomic(
                    path,
                    {
                        "version": 1,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "geo": geo,
                    },
                )
            if getattr(self, "_geo_status_var", None) is not None:
                self._geo_status_var.set(self._format_geo_status())
        except Exception:
            pass

    def _clear_geo_cache(self):
        try:
            path = self._geo_state_path()
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            if getattr(self, "_geo_status_var", None) is not None:
                self._geo_status_var.set(self._format_geo_status())
        except Exception:
            pass

    def _install_context_menus(self):
        self._menu_input = tk.Menu(self, tearoff=0)
        self._menu_input.add_command(label="Вырезать", command=lambda: self.entry.event_generate("<<Cut>>"))
        self._menu_input.add_command(label="Копировать", command=lambda: self.entry.event_generate("<<Copy>>"))
        self._menu_input.add_command(label="Вставить", command=lambda: self.entry.event_generate("<<Paste>>"))
        self._menu_input.add_separator()
        self._menu_input.add_command(label="Выделить всё", command=lambda: self._select_all(self.entry))

        def popup_input(e):
            try:
                self.entry.focus_set()
                self._menu_input.tk_popup(e.x_root, e.y_root)
            finally:
                try:
                    self._menu_input.grab_release()
                except Exception:
                    pass

        self.entry.bind("<Button-3>", popup_input)
        self.entry.bind("<Button-2>", popup_input)

        self.entry.bind("<Control-a>", lambda e: (self._select_all(self.entry), "break")[1])
        self.entry.bind("<Control-A>", lambda e: (self._select_all(self.entry), "break")[1])

        self._menu_chat = tk.Menu(self, tearoff=0)
        self._menu_chat.add_command(label="Копировать", command=lambda: self._copy_selection(self.chat))
        self._menu_chat.add_separator()
        self._menu_chat.add_command(label="Выделить всё", command=lambda: self._select_all(self.chat))

        def popup_chat(e):
            try:
                self.chat.focus_set()
                self._menu_chat.tk_popup(e.x_root, e.y_root)
            finally:
                try:
                    self._menu_chat.grab_release()
                except Exception:
                    pass

        self.chat.bind("<Button-3>", popup_chat)
        self.chat.bind("<Button-2>", popup_chat)
        self.chat.bind("<Button-1>", lambda _e: self.chat.focus_set())

        self.chat.bind("<Control-c>", lambda e: (self._copy_selection(self.chat), "break")[1])
        self.chat.bind("<Control-C>", lambda e: (self._copy_selection(self.chat), "break")[1])
        self.chat.bind("<Control-a>", lambda e: (self._select_all(self.chat), "break")[1])
        self.chat.bind("<Control-A>", lambda e: (self._select_all(self.chat), "break")[1])

    def _select_all(self, widget: tk.Text):
        try:
            widget.tag_add("sel", "1.0", "end-1c")
            widget.mark_set("insert", "1.0")
            widget.see("insert")
        except Exception:
            pass

    def _copy_selection(self, widget: tk.Text):
        try:
            sel = widget.get("sel.first", "sel.last")
        except Exception:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(sel)
        except Exception:
            pass

    def _on_send(self):
        user_text = self.entry.get("1.0", "end-1c").strip()
        if not user_text:
            return

        self._clear_input()
        self._add_message("user", "Ты", user_text)
        if self._attached_files:
            names = [os.path.basename(p) or "file" for p in self._attached_files]
            self._add_message("meta", "Ты", "Прикреплено файлов: " + ", ".join(names), include_in_context=False)

        self._show_typing()

        payload = self._build_task_payload(user_text)

        self._polling = False
        self._poll_inflight = False
        self._last_task_status = ""
        self.current_task_id = None

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

        if self._sending and self.current_task_id is None and self._sending_started_at:
            if (time.time() - float(self._sending_started_at)) > SEND_TIMEOUT_SEC:
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

        final_answer = ""
        if isinstance(result, dict):
            final_answer = str(result.get("final_answer") or "").strip()

        if final_answer:
            self._polling = False
            self._clear_typing_if_any()
            self._add_message("assistant", AA_DISPLAY_NAME, final_answer)
            self._announce_task_files_in_chat(task_id, task=task)
            return

        if status in {"DONE", "REVIEW_NEEDS_ATTENTION"}:
            self._polling = False
            self._clear_typing_if_any()
            msg = "Задача завершилась, но финальный ответ пуст. Открой «Логи» и пришли скрин последних строк."
            self._add_message("assistant", AA_DISPLAY_NAME, msg)
            return

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

    def _open_task_files(self, files: list[dict] | None = None):
        if self.current_task_id is None:
            messagebox.showinfo("Скачать", "Сначала дождись задачи с файлами.")
            return

        try:
            files = files if files is not None else self.client.list_task_files(self.current_task_id)
        except Exception as e:
            messagebox.showerror("Скачать", f"Не удалось получить список файлов: {e}")
            return

        if not files:
            messagebox.showinfo("Скачать", "Для этой задачи файлов пока нет.")
            return

        win = tk.Toplevel(self)
        win.title(f"Файлы задачи #{self.current_task_id}")
        win.geometry("760x420")

        btns = tk.Frame(win, padx=10, pady=(10, 10))
        btns.pack(side=tk.BOTTOM, fill=tk.X)

        frm = tk.Frame(win, padx=10, pady=(10, 0))
        frm.pack(fill=tk.BOTH, expand=True)

        lb = tk.Listbox(frm, font=("Segoe UI", 10))
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sc = tk.Scrollbar(frm, command=lb.yview)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        lb.configure(yscrollcommand=sc.set)

        file_map: list[dict] = []
        for item in files:
            file_id = int(item.get("id") or 0)
            file_name = str(item.get("file_name") or f"file_{file_id}")
            size_bytes = int(item.get("size_bytes") or 0)
            label = f"{file_name}  ({size_bytes} bytes)"
            lb.insert(tk.END, label)
            file_map.append(item)

        def selected_items() -> list[dict]:
            sels = lb.curselection()
            return [file_map[int(i)] for i in sels] if sels else []

        def do_download_selected():
            items = selected_items()
            if not items:
                messagebox.showinfo("Скачать", "Выбери файл в списке.", parent=win)
                return
            if len(items) == 1:
                self._save_task_file(self.current_task_id, items[0], parent=win)
                return

            target_dir = filedialog.askdirectory(title="Куда сохранить файлы", parent=win)
            if not target_dir:
                return

            saved = 0
            for meta in items:
                file_id = int(meta.get("id") or 0)
                file_name = str(meta.get("file_name") or f"file_{file_id}")
                save_path = os.path.join(target_dir, file_name)
                try:
                    content = self.client.download_task_file(self.current_task_id, file_id)
                    with open(save_path, "wb") as f:
                        f.write(content or b"")
                    saved += 1
                except Exception as e:
                    messagebox.showerror("Скачать", f"Не удалось скачать файл {file_name}: {e}", parent=win)
                    return

            messagebox.showinfo("Скачать", f"Сохранено файлов: {saved}\nПапка: {target_dir}", parent=win)

        def do_download_all():
            lb.selection_clear(0, tk.END)
            lb.selection_set(0, tk.END)
            do_download_selected()

        tk.Button(btns, text="Скачать выбранное", command=do_download_selected).pack(side=tk.LEFT)
        tk.Button(btns, text="Скачать всё", command=do_download_all).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(btns, text="Обновить", command=lambda: (win.destroy(), self._open_task_files())).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(btns, text="Закрыть", command=win.destroy).pack(side=tk.RIGHT)

        lb.bind("<Double-Button-1>", lambda _e: do_download_selected())
        lb.bind("<Return>", lambda _e: do_download_selected())

    def _open_logs(self):
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
        win.geometry("1100x650")

        body = tk.Frame(win)
        body.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(body, wrap="none", font=("Consolas", 10))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sc_y = tk.Scrollbar(body, command=txt.yview)
        sc_y.pack(side=tk.RIGHT, fill=tk.Y)
        sc_x = tk.Scrollbar(win, orient=tk.HORIZONTAL, command=txt.xview)
        sc_x.pack(side=tk.BOTTOM, fill=tk.X)
        txt.configure(yscrollcommand=sc_y.set, xscrollcommand=sc_x.set)

        if not logs:
            txt.insert(tk.END, "Логи не найдены.\n")

        for item in logs:
            ts = item.get("created_at") or ""
            actor = item.get("actor_agent") or ""
            ev = item.get("event_type") or ""
            lvl = item.get("level") or ""
            msg = item.get("message") or ""
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}

            txt.insert(tk.END, f"{ts} [{lvl}] {actor} {ev} — {msg}\n")
            if meta:
                meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
                for line in meta_text.splitlines():
                    txt.insert(tk.END, f"    {line}\n")
            txt.insert(tk.END, "\n")

        txt.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
