# -*- coding: utf-8 -*-
"""
Ozonator Agents Client (.pyw) — single file, no external deps.

Назначение:
- В одном окне: поставить задачу -> AA запускает цепочку -> получить финальный ответ (task.result.final_answer).

AA API (основные пути):
- POST /tasks/create
- POST /aa/run-task/{task_id}
- GET  /tasks/{task_id}
- GET  /tasks/{task_id}/logs

Примечания:
- База AA берётся из OZONATOR_AA_BASE_URL (по умолчанию https://ozonator-aa-dev.onrender.com)
- Токены НЕ хранятся в коде:
  - OZONATOR_AA_BEARER (Authorization: Bearer ...)
  - OZONATOR_AA_ADMIN_TOKEN (X-Admin-Token ...)
- Клиент старается переживать разные префиксы деплоя (/api, /v1) и варианты путей.
- Таймаут вызова /aa/run-task может происходить на cold-start — это НЕ ошибка: клиент продолжает опрашивать /tasks/{id}.
"""

import json
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont


# =========================
# Config (hidden)
# =========================
DEFAULT_BASE_URL = os.environ.get("OZONATOR_AA_BASE_URL", "https://ozonator-aa-dev.onrender.com").rstrip("/")
AA_BEARER = os.environ.get("OZONATOR_AA_BEARER", "").strip()
AA_ADMIN_TOKEN = os.environ.get("OZONATOR_AA_ADMIN_TOKEN", "").strip()

DEFAULT_LLM_PROVIDER = os.environ.get("OZONATOR_LLM_PROVIDER", "groq").strip() or "groq"
DEFAULT_LLM_MODEL = os.environ.get("OZONATOR_LLM_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"

APP_NAME = "Ozonator Agents Client"
AA_DISPLAY_NAME = "Екатерина"
SHOW_HISTORY_PANEL = False  # скрыть левую панель истории диалогов

HISTORY_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OzonatorAgentsClient"
HISTORY_FILE = HISTORY_DIR / "history.json"
LOG_FILE = HISTORY_DIR / "client.log"

CONVERSATION_FILE = HISTORY_DIR / "conversation.json"
CONVERSATION_MAX_MESSAGES = 200
CONTEXT_MAX_MESSAGES = 18
CONTEXT_MAX_CHARS = 3200


POLL_INTERVAL_SEC = 1.0
HTTP_TIMEOUT_SEC = 60
RUN_TASK_TIMEOUT_SEC = 10  # короткий "пинок" оркестрации — дальше работаем polling'ом
CREATE_RETRIES = 4

# Telegram-like UI palette
TG_BG = "#0f1720"
TG_PANEL = "#17212b"
TG_PANEL_ALT = "#1f2c3a"
TG_HEADER = "#182533"
TG_BORDER = "#22303d"
TG_TEXT = "#e6edf3"
TG_MUTED = "#8ea2b5"
TG_ACCENT = "#2aabee"
TG_ACCENT_HOVER = "#48b8f2"
TG_ME_BG = "#2b5278"
TG_AA_BG = "#1f2c3a"
TG_SYSTEM_BG = "#22303d"
TG_INPUT_BG = "#17212b"

# Embedded Telegram-style avatar for Екатерина (circular PNG, base64)
AVATAR_PNG_B64 = '''
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
KKsMpRX/B2UX0kiR5kJEAAAAAElFTkSuQmCC
'''



# =========================
# Helpers
# =========================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_log(msg: str) -> None:
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{_ts()}] {msg}\n")
    except Exception:
        pass


def short_title(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return (t[:80] + "…") if len(t) > 80 else t


STATUS_RU = {
    "NEW": "Создано",
    "IN_PROGRESS": "В работе",
    "AA_ROUTED": "Передано в контур",
    "BRIEF_READY": "Постановка готова",
    "ARTIFACTS_READY": "Артефакты готовы",
    "REVIEW_NEEDS_ATTENTION": "Нужны исправления",
    "DONE": "Готово",
    "FAILED": "Ошибка",
    "CANCELLED": "Отменено",
}


def status_ru(status: Any) -> str:
    if not status:
        return "Неизвестный статус"
    s = str(status).upper()
    return STATUS_RU.get(s, f"Неизвестный статус: {status}")


def _json_load(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _json_save(path: Path, obj: Any) -> None:
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _strip_bearer(s: str) -> str:
    s = (s or "").strip()
    if s.lower().startswith("bearer "):
        return s.split(" ", 1)[1].strip()
    return s


# =========================
# Geo + Conversation Context
# =========================
def _safe_get_json(url: str, timeout_sec: int = 5) -> dict:
    # Simple JSON GET without auth headers (used for public geo endpoints)
    try:
        from urllib import request as _rq
        req = _rq.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (OzonatorAgentsClient)",
            },
        )
        with _rq.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            import json as _json
            return _json.loads(raw)
    except Exception:
        return {}


def detect_geo_context() -> dict:
    # Best-effort geo by public IP. Returns {} on any failure.
    data = _safe_get_json("https://ipapi.co/json/", timeout_sec=5)
    if isinstance(data, dict) and data:
        city = (data.get("city") or "").strip()
        region = (data.get("region") or data.get("region_code") or "").strip()
        country = (data.get("country_name") or data.get("country") or "").strip()
        lat = data.get("latitude")
        lon = data.get("longitude")
        tz = (data.get("timezone") or "").strip()
        ctx = {
            "city": city,
            "region": region,
            "country": country,
            "lat": lat,
            "lon": lon,
            "timezone": tz,
        }
        return {k: v for k, v in ctx.items() if v not in (None, "")}

    data = _safe_get_json("https://ipinfo.io/json", timeout_sec=5)
    if isinstance(data, dict) and data:
        loc = (data.get("loc") or "").strip()
        lat = lon = None
        if "," in loc:
            try:
                lat_s, lon_s = loc.split(",", 1)
                lat = float(lat_s.strip())
                lon = float(lon_s.strip())
            except Exception:
                pass
        city = (data.get("city") or "").strip()
        region = (data.get("region") or "").strip()
        country = (data.get("country") or "").strip()
        ctx = {
            "city": city,
            "region": region,
            "country": country,
            "lat": lat,
            "lon": lon,
        }
        return {k: v for k, v in ctx.items() if v not in (None, "")}

    return {}


def build_context_block(messages: list, geo: dict) -> str:
    # Render recent dialogue + geo into a compact prompt block.
    parts = []

    if isinstance(geo, dict) and geo:
        city = geo.get("city")
        region = geo.get("region")
        country = geo.get("country")
        tz = geo.get("timezone")
        loc_bits = [b for b in [city, region, country] if b]
        if loc_bits:
            loc_line = ", ".join(loc_bits)
            if tz:
                loc_line += f" (TZ: {tz})"
            parts.append(f"Локация (авто, по IP): {loc_line}.")

    if messages:
        parts.append("Контекст (последние сообщения):")
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            txt = (m.get("text") or "").strip()
            if not txt:
                continue
            if role == "user":
                parts.append(f"- Вы: {txt}")
            elif role == "assistant":
                parts.append(f"- AA: {txt}")

    parts.append(
        "Требования к ответу: "
        "если запрос неоднозначен — предложи 2–3 трактовки и ответь по наиболее вероятной; "
        "задай максимум 1 уточняющий вопрос только если без него нельзя; "
        "не пиши шаблонно ‘мне не хватает информации’ без попытки помочь; "
        "не выводи IP/координаты, если пользователь не просит."
    )

    return "\n".join([p for p in parts if (p or "").strip()]).strip()


# =========================
# HTTP JSON
# =========================
def http_json(method: str, url: str, payload: Optional[dict] = None, timeout_sec: int = HTTP_TIMEOUT_SEC) -> Tuple[int, Dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (OzonatorAgentsClient)",
    }
    if AA_BEARER:
        headers["Authorization"] = f"Bearer {_strip_bearer(AA_BEARER)}"
    if AA_ADMIN_TOKEN:
        headers["X-Admin-Token"] = AA_ADMIN_TOKEN

    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = request.Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return resp.status, {}
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, {"detail": raw.strip()}
    except error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        try:
            j = json.loads(raw) if raw.strip() else {}
        except Exception:
            j = {"detail": raw.strip()} if raw.strip() else {}
        return int(getattr(e, "code", 0) or 0), j
    except error.URLError as e:
        reason = getattr(e, "reason", e)
        return 0, {"detail": f"URLError: {reason}"}
    except Exception as e:
        return 0, {"detail": f"{e.__class__.__name__}: {e}"}


# =========================
# Extractors
# =========================
def extract_final_answer(task_obj: Dict[str, Any]) -> str:
    if not isinstance(task_obj, dict):
        return ""
    result = task_obj.get("result") if isinstance(task_obj.get("result"), dict) else None
    if result and isinstance(result.get("final_answer"), str):
        return result.get("final_answer") or ""
    payload = task_obj.get("payload") if isinstance(task_obj.get("payload"), dict) else None
    if payload and isinstance(payload.get("final_answer"), str):
        return payload.get("final_answer") or ""
    # fallback fields
    for key in ("final_answer", "answer", "output"):
        v = task_obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_question_to_user(task_obj: Dict[str, Any]) -> str:
    if not isinstance(task_obj, dict):
        return ""
    # Common patterns
    for path in [
        ("result", "question_to_user"),
        ("result", "next_question"),
        ("result", "user_question"),
        ("payload", "question_to_user"),
        ("payload", "next_question"),
        ("payload", "user_question"),
        ("question_to_user",),
        ("next_question",),
        ("user_question",),
    ]:
        cur = task_obj
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()

    # list of questions
    res = task_obj.get("result")
    if isinstance(res, dict):
        qs = res.get("questions")
        if isinstance(qs, list) and qs:
            joined = "\n".join([str(x) for x in qs if str(x).strip()])
            return joined.strip()
    return ""


# =========================
# AA API Client
# =========================
class AAClient:
    def __init__(self, base_url: str):
        self.base_url = (base_url or "").rstrip("/")
        self.api_prefix = ""  # '', '/api', '/v1'

    def _mk(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    def _prefixes(self) -> List[str]:
        return ["", "/api", "/v1"]

    def create_task(self, user_text: str) -> int:
        body = {
            "target_agent": "AA",
            "task_type": "user_task",
            "priority": 100,
            "payload": {
                "user_request": user_text,
                "llm_provider": DEFAULT_LLM_PROVIDER,
                "llm_model": DEFAULT_LLM_MODEL,
                "client_meta": {"source": "desktop", "client": "OzonatorAgentsClient"},
            },
        }

        last_err = None
        # cold-start friendly retries
        for attempt in range(CREATE_RETRIES):
            for pref in self._prefixes():
                self.api_prefix = pref
                for url in (self._mk("/tasks/create"), self._mk("/tasks"), self._mk("/task/create")):
                    code, resp = http_json("POST", url, body, timeout_sec=HTTP_TIMEOUT_SEC)
                    if code == 200:
                        if isinstance(resp, dict):
                            task = resp.get("task")
                            if isinstance(task, dict) and "id" in task:
                                return int(task["id"])
                            if "id" in resp:
                                return int(resp["id"])
                        # fallback
                        raise RuntimeError(f"create_task: не нашёл id в ответе: {resp}")
                    if code in (404, 405):
                        last_err = (code, resp, url)
                        continue
                    if code == 0:
                        last_err = (code, resp, url)
                        continue
                    # real error
                    raise RuntimeError(f"create_task: HTTP {code}: {resp}")
            # wait and retry
            time.sleep(1.0 + attempt * 0.8)
        raise RuntimeError(f"create_task: не удалось. Последняя ошибка: {last_err}")

    def run_task_kick(self, task_id: int) -> None:
        """
        Запуск оркестрации. На cold-start может не ответить в разумный срок — это не фатально.
        """
        tried = set()
        for pref in [self.api_prefix] + [p for p in self._prefixes() if p != self.api_prefix]:
            self.api_prefix = pref
            for url in (
                self._mk(f"/aa/run-task/{task_id}"),
                self._mk(f"/aa/run/{task_id}"),
                self._mk(f"/tasks/{task_id}/run"),
                self._mk(f"/tasks/run/{task_id}"),
            ):
                if url in tried:
                    continue
                tried.add(url)
                code, resp = http_json("POST", url, None, timeout_sec=RUN_TASK_TIMEOUT_SEC)
                if code == 200:
                    return
                # 0 => timeout/URL error; 404/405 => попробуем другие варианты; остальное — логируем, но не падаем
                if code in (0, 404, 405):
                    continue
                safe_log(f"run_task_kick: HTTP {code} at {url}: {resp}")
                return

    def get_task(self, task_id: int) -> dict:
        last_err = None
        tried = set()
        for pref in [self.api_prefix] + [p for p in self._prefixes() if p != self.api_prefix]:
            self.api_prefix = pref
            for url in (
                self._mk(f"/tasks/{task_id}"),
                self._mk(f"/task/{task_id}"),
            ):
                if url in tried:
                    continue
                tried.add(url)
                code, resp = http_json("GET", url, None)
                if code == 200:
                    if isinstance(resp, dict) and "task" in resp and isinstance(resp["task"], dict):
                        return resp["task"]
                    return resp if isinstance(resp, dict) else {}
                if code in (404, 405):
                    last_err = (code, resp, url)
                    continue
                # keep last error
                last_err = (code, resp, url)
        raise RuntimeError(f"get_task: не удалось. Последняя ошибка: {last_err}")

    def get_logs(self, task_id: int) -> List[str]:
        tried = set()
        for pref in [self.api_prefix] + [p for p in self._prefixes() if p != self.api_prefix]:
            self.api_prefix = pref
            for url in (
                self._mk(f"/tasks/{task_id}/logs"),
                self._mk(f"/task/{task_id}/logs"),
                self._mk(f"/logs/{task_id}"),
            ):
                if url in tried:
                    continue
                tried.add(url)
                code, resp = http_json("GET", url, None)
                if code == 200:
                    if isinstance(resp, dict):
                        logs = resp.get("logs") or resp.get("items") or resp.get("data")
                        if isinstance(logs, list):
                            return [str(x) for x in logs]
                        if isinstance(resp.get("text"), str):
                            return [resp["text"]]
                    if isinstance(resp, list):
                        return [str(x) for x in resp]
                    return [json.dumps(resp, ensure_ascii=False)]
                if code in (404, 405):
                    continue
        return []


# =========================
# UI data
# =========================

class RoundIconButton(tk.Canvas):
    """Telegram-like round icon button (no external deps)."""

    def __init__(
        self,
        master,
        diameter: int = 44,
        text: str = "➤",
        command=None,
        bg_color: str = TG_ACCENT,
        hover_color: str = TG_ACCENT_HOVER,
        text_color: str = "#ffffff",
        font_obj=None,
        **kwargs,
    ):
        super().__init__(
            master,
            width=diameter,
            height=diameter,
            bg=master.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            **kwargs,
        )
        self._diameter = diameter
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._command = command

        self._oval = self.create_oval(2, 2, diameter - 2, diameter - 2, fill=bg_color, outline="")
        self._txt = self.create_text(
            diameter // 2,
            diameter // 2,
            text=text,
            fill=text_color,
            font=font_obj,
        )

        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        # make text clickable too
        self.tag_bind(self._txt, "<Button-1>", self._on_click)
        self.tag_bind(self._txt, "<Enter>", self._on_enter)
        self.tag_bind(self._txt, "<Leave>", self._on_leave)

    def _on_click(self, _evt=None):
        if callable(self._command):
            self._command()

    def _on_enter(self, _evt=None):
        self.itemconfigure(self._oval, fill=self._hover_color)

    def _on_leave(self, _evt=None):
        self.itemconfigure(self._oval, fill=self._bg_color)


@dataclass
class TaskItem:
    id: int
    title: str
    created_at: str
    base_url: str
    api_prefix: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "base_url": self.base_url,
            "api_prefix": self.api_prefix,
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskItem":
        return TaskItem(
            id=int(d.get("id")),
            title=str(d.get("title") or ""),
            created_at=str(d.get("created_at") or ""),
            base_url=str(d.get("base_url") or DEFAULT_BASE_URL),
            api_prefix=str(d.get("api_prefix") or ""),
        )


# =========================
# App
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1280x800")
        self.minsize(1040, 680)
        self.configure(bg=TG_BG)
        try:
            self.state("zoomed")
        except Exception:
            pass
        self._configure_theme()

        self.q = queue.Queue()
        self.client = AAClient(DEFAULT_BASE_URL)

        self.items: List[TaskItem] = []
        self.current_task_id: Optional[int] = None
        self.last_status: Optional[str] = None
        self._polling = False
        self._stop = False

        self._load_history()
        self._load_conversation()
        self.geo_context: dict = {}
        self._start_geo_detection()
        self._build_ui()
        self._refresh_history_list()

        self.after(200, self._drain_queue)
        self.after(int(POLL_INTERVAL_SEC * 1000), self._tick)
        self.after(1500, self._keep_warm_tick)

    def _configure_theme(self):
        self.font_ui_8 = tkfont.Font(family="Segoe UI", size=8)
        self.font_ui_9 = tkfont.Font(family="Segoe UI", size=9)
        self.font_ui_10 = tkfont.Font(family="Segoe UI", size=10)
        self.font_ui_9_bold = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.font_ui_10_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_ui_11_bold = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.font_ui_12_bold = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.font_mono_10 = tkfont.Font(family="Consolas", size=10)
        self.option_add("*Font", self.font_ui_10)
        self.option_add("*Menu.background", TG_PANEL)
        self.option_add("*Menu.foreground", TG_TEXT)
        self.option_add("*Menu.activeBackground", TG_ACCENT)
        self.option_add("*Menu.activeForeground", "#ffffff")

    def _build_ui(self):
        header = tk.Frame(self, bg=TG_HEADER, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        # Avatar (circle) like in Telegram
        try:
            self.avatar_img = tk.PhotoImage(data="".join(AVATAR_PNG_B64.split()))
        except Exception:
            self.avatar_img = None

        avatar = tk.Label(
            header,
            image=self.avatar_img,
            bg=TG_HEADER,
            bd=0,
        )
        avatar.pack(side="left", padx=(16, 12), pady=10)

        title_wrap = tk.Frame(header, bg=TG_HEADER)
        title_wrap.pack(side="left", fill="y", pady=10)
        tk.Label(
            title_wrap,
            text=AA_DISPLAY_NAME,
            bg=TG_HEADER,
            fg=TG_TEXT,
            font=self.font_ui_12_bold,
        ).pack(anchor="w")
        tk.Label(
            title_wrap,
            text="● online",
            bg=TG_HEADER,
            fg="#4cd964",
            font=self.font_ui_9,
        ).pack(anchor="w", pady=(2, 0))

        tk.Button(
            header,
            text="≡ Логи",
            command=self._show_logs,
            bg=TG_PANEL_ALT,
            fg=TG_TEXT,
            activebackground=TG_BORDER,
            activeforeground=TG_TEXT,
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            font=self.font_ui_9_bold,
        ).pack(side="right", padx=16, pady=12)

        body = tk.Frame(self, bg=TG_BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)

        # По НФ: левую панель истории чатов скрываем.
        self.lb = None
        right = tk.Frame(body, bg=TG_BG)
        right.pack(side="left", fill="both", expand=True)
        # (UI) Верхнюю строку статуса убрали по ТЗ.
        # status_var оставляем для внутренних обновлений/логов.
        self.status_var = tk.StringVar(value="")

        chat_wrap = tk.Frame(right, bg=TG_BG, bd=0, highlightthickness=1, highlightbackground=TG_BORDER)
        chat_wrap.pack(fill="both", expand=True)

        self.chat = tk.Text(
            chat_wrap,
            wrap="word",
            height=20,
            bg=TG_BG,
            fg=TG_TEXT,
            insertbackground=TG_TEXT,
            selectbackground=TG_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=12,
            font=self.font_ui_10,
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.configure(state="disabled")

        # По НФ: убираем «зебру» — одинаковый фон, без заливки строк.
        self.chat.tag_configure(
            "aa_ts",
            foreground=TG_MUTED,
            font=self.font_ui_8,
            lmargin1=18,
            lmargin2=18,
            rmargin=170,
            spacing1=8,
        )
        self.chat.tag_configure(
            "user_ts",
            foreground=TG_MUTED,
            font=self.font_ui_8,
            justify="right",
            lmargin1=170,
            lmargin2=170,
            rmargin=18,
            spacing1=8,
        )
        self.chat.tag_configure(
            "system_ts",
            foreground=TG_MUTED,
            font=self.font_ui_8,
            justify="center",
            lmargin1=120,
            lmargin2=120,
            rmargin=120,
            spacing1=8,
        )

        # Важно: background НЕ задаём — так исчезают «полосы».
        self.chat.tag_configure("aa_msg", foreground=TG_TEXT, lmargin1=18, lmargin2=18, rmargin=170, spacing3=8)
        self.chat.tag_configure("user_msg", foreground="#ffffff", justify="right", lmargin1=170, lmargin2=170, rmargin=18, spacing3=8)
        self.chat.tag_configure("system_msg", foreground=TG_TEXT, justify="center", lmargin1=120, lmargin2=120, rmargin=120, spacing3=8)

        bottom = tk.Frame(right, bg=TG_PANEL, bd=0, highlightthickness=1, highlightbackground=TG_BORDER)
        bottom.pack(fill="x", pady=(12, 0))

        self.input = tk.Text(
            bottom,
            wrap="word",
            height=4,
            bg=TG_INPUT_BG,
            fg=TG_TEXT,
            insertbackground=TG_TEXT,
            selectbackground=TG_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=self.font_ui_10,
        )
        self.input.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        # Enter: send, Shift+Enter: new line
        self.input.bind('<Return>', self._on_input_enter)
        self.input.bind('<KP_Enter>', self._on_input_enter)
        self.input.bind('<Shift-Return>', self._on_input_shift_enter)
        self.input.bind('<Shift-KP_Enter>', self._on_input_shift_enter)

        actions = tk.Frame(bottom, bg=TG_PANEL)
        actions.pack(side="left", padx=(0, 10), pady=8)

        # Telegram-like send button (round icon).
        self.send_btn = RoundIconButton(
            actions,
            diameter=46,
            text="➤",
            command=self._send,
            bg_color=TG_ACCENT,
            hover_color=TG_ACCENT_HOVER,
            font_obj=self.font_ui_12_bold,
        )
        self.send_btn.pack()

        # Input clear button removed per UI request.
        # Shortcut: Ctrl+L clears input.
        self.input.bind('<Control-l>', lambda e: (self._clear_input(), 'break'))


    # ---------- history ----------

    # ---------- history ----------
    def _load_history(self):
        data = _json_load(HISTORY_FILE, [])
        self.items = []
        if isinstance(data, list):
            for d in data:
                try:
                    self.items.append(TaskItem.from_dict(d))
                except Exception:
                    continue

    def _save_history(self):
        _json_save(HISTORY_FILE, [i.to_dict() for i in self.items])

    def _refresh_history_list(self):
        if getattr(self, 'lb', None) is None:
            return
        self.lb.delete(0, tk.END)
        for it in self.items:
            self.lb.insert(tk.END, f"{it.title} (#{it.id})")

    def _on_select_history(self, _evt=None):
        lb = getattr(self, 'lb', None)
        if lb is None:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.items):
            return
        it = self.items[idx]
        self.current_task_id = it.id
        self.client.base_url = it.base_url
        self.client.api_prefix = it.api_prefix or ""
        self._append_system(f"Выбрана задача #{it.id}.")
        self._start_polling()

    def _delete_selected(self):
        lb = getattr(self, 'lb', None)
        if lb is None:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.items):
            return
        it = self.items.pop(idx)
        self._save_history()
        self._refresh_history_list()
        if self.current_task_id == it.id:
            self.current_task_id = None
        self._append_system(f"Удалено из истории: #{it.id}.")

    def _clear_history(self):
        if not self.items:
            return
        if messagebox.askyesno("Подтверждение", "Очистить всю историю?"):
            self.items = []
            self._save_history()
            self._refresh_history_list()
            self.current_task_id = None
            self._append_system("История очищена.")

    # ---------- chat ----------
    def _append(self, who: str, text: str):
        text = (text or "").strip()
        if not text:
            return

        stamp = datetime.now().strftime("%H:%M")
        if who == "Вы":
            ts_tag, msg_tag = "user_ts", "user_msg"
        elif who in ("AA", AA_DISPLAY_NAME):
            ts_tag, msg_tag = "aa_ts", "aa_msg"
        else:
            ts_tag, msg_tag = "system_ts", "system_msg"

        self.chat.configure(state="normal")
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert(tk.END, "\n")
        self.chat.insert(tk.END, f"{stamp}\n", (ts_tag,))
        self.chat.insert(tk.END, f"  {text}  \n", (msg_tag,))
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

        try:
            self._remember_message(who, text)
        except Exception:
            pass

    def _append_system(self, text: str):
        self._append("Система", text)

    def _append_user(self, text: str):
        self._append("Вы", text)

    def _append_aa(self, text: str):
        self._append(AA_DISPLAY_NAME, text)

    # ---------- conversation ----------
    def _load_conversation(self):
        data = _json_load(CONVERSATION_FILE, [])
        self.conversation: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for m in data:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip()
                txt = str(m.get("text") or "").strip()
                if role in ("user", "assistant") and txt:
                    self.conversation.append({"role": role, "text": txt, "ts": str(m.get("ts") or "")})
        if len(self.conversation) > CONVERSATION_MAX_MESSAGES:
            self.conversation = self.conversation[-CONVERSATION_MAX_MESSAGES:]

    def _save_conversation(self):
        _json_save(CONVERSATION_FILE, self.conversation[-CONVERSATION_MAX_MESSAGES:])

    def _remember_message(self, who: str, text: str):
        who = (who or "").strip()
        txt = (text or "").strip()
        if not txt:
            return
        role = None
        if who == "Вы":
            role = "user"
        elif who in ("AA", AA_DISPLAY_NAME):
            role = "assistant"
        else:
            return
        self.conversation.append({"role": role, "text": txt, "ts": datetime.now().isoformat(timespec="seconds")})
        if len(self.conversation) > CONVERSATION_MAX_MESSAGES:
            self.conversation = self.conversation[-CONVERSATION_MAX_MESSAGES:]
        self._save_conversation()

    def _recent_context_messages(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        total = 0
        for m in reversed(self.conversation):
            if len(out) >= CONTEXT_MAX_MESSAGES:
                break
            t = (m.get("text") or "").strip()
            if not t:
                continue
            add = len(t) + 20
            if total + add > CONTEXT_MAX_CHARS and out:
                break
            out.append({"role": m.get("role"), "text": t})
            total += add
        out.reverse()
        return out

    def _compose_user_request(self, user_text: str) -> str:
        ctx_msgs = self._recent_context_messages()
        ctx_block = build_context_block(ctx_msgs, getattr(self, "geo_context", {}) or {})
        user_text = (user_text or "").strip()
        if ctx_block:
            return f"{ctx_block}\n\nЗАПРОС:\n{user_text}".strip()
        return user_text

    def _start_geo_detection(self):
        def worker():
            try:
                geo = detect_geo_context()
                if isinstance(geo, dict) and geo:
                    self.geo_context = geo
                    safe_log(f"geo_context detected: {geo}")
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _keep_warm_tick(self):
        def worker():
            try:
                base = (self.client.base_url or "").rstrip("/")
                if not base:
                    return
                for path in ("/health", "/docs", "/"):
                    try:
                        code, _ = http_json("GET", base + path, None, timeout_sec=3)
                        if code in (200, 401, 403):
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()
        if not getattr(self, "_stop", False):
            self.after(55000, self._keep_warm_tick)

    def _should_hide_aa_line(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if t in ("принято. выполняю.", "принято. выполняю", "принято — выполняю.", "принято — выполняю"):
            return True
        return False

    # ---------- actions ----------
    def _on_input_enter(self, _evt=None):
        # Send on Enter
        self._send()
        return 'break'

    def _on_input_shift_enter(self, _evt=None):
        # New line on Shift+Enter
        self.input.insert(tk.INSERT, '\n')
        return 'break'

    def _clear_input(self):
        self.input.delete("1.0", tk.END)

    def _send(self):
        user_text = self.input.get("1.0", tk.END).strip()
        if not user_text:
            return
        self._clear_input()
        self._append_user(user_text)
        self.status_var.set("Отправка задачи…")

        def worker():
            try:
                composed = self._compose_user_request(user_text)
                task_id = self.client.create_task(composed)
                # store task
                it = TaskItem(
                    id=task_id,
                    title=short_title(user_text),
                    created_at=_ts(),
                    base_url=self.client.base_url,
                    api_prefix=self.client.api_prefix,
                )
                # move to top, unique
                self.items = [x for x in self.items if x.id != task_id]
                self.items.insert(0, it)
                self._save_history()

                self.q.put(("task_created", task_id, it))
                # kick orchestration (non-fatal)
                try:
                    self.client.run_task_kick(task_id)
                except Exception as e:
                    safe_log(f"run_task_kick exception: {e}")
                self.q.put(("kick_done", task_id, None))
            except Exception as e:
                self.q.put(("error", "Не удалось отправить задачу", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _start_polling(self):
        if self._polling or self.current_task_id is None:
            return
        self._polling = True
        self.last_status = None

    def _tick(self):
        if self._stop:
            return
        try:
            if self._polling and self.current_task_id is not None:
                def poll_worker(task_id: int):
                    try:
                        task = self.client.get_task(task_id)
                        self.q.put(("task_update", task_id, task))
                    except Exception as e:
                        self.q.put(("poll_error", task_id, str(e)))
                threading.Thread(target=poll_worker, args=(self.current_task_id,), daemon=True).start()
        finally:
            self.after(int(POLL_INTERVAL_SEC * 1000), self._tick)

    def _show_logs(self):
        if self.current_task_id is None:
            # show local log
            txt = ""
            try:
                txt = LOG_FILE.read_text(encoding="utf-8", errors="replace") if LOG_FILE.exists() else ""
            except Exception:
                txt = ""
            if not txt.strip():
                messagebox.showinfo("Логи", "Логов нет.")
                return
            self._popup_text("Логи клиента", txt)
            return

        task_id = self.current_task_id
        self.status_var.set(f"Загрузка логов задачи #{task_id}…")

        def worker():
            try:
                logs = self.client.get_logs(task_id)
                txt = "\n".join(logs) if logs else "Логов нет."
                self.q.put(("show_logs", task_id, txt))
            except Exception as e:
                self.q.put(("error", "Не удалось загрузить логи", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _popup_text(self, title: str, text: str):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("960x640")
        win.configure(bg=TG_BG)
        frame = tk.Frame(win, bg=TG_BG, padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        t = tk.Text(
            frame,
            wrap="word",
            bg=TG_PANEL,
            fg=TG_TEXT,
            insertbackground=TG_TEXT,
            selectbackground=TG_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            font=self.font_mono_10,
        )
        t.pack(fill="both", expand=True)
        t.insert("1.0", text)
        t.configure(state="disabled")

    # ---------- queue ----------
    def _drain_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(200, self._drain_queue)

    def _handle_msg(self, msg):
        kind = msg[0]
        if kind == "error":
            _, title, detail = msg
            self.status_var.set("Ошибка")
            self._append("Ошибка", f"{title}: {detail}")
            return

        if kind == "task_created":
            _, task_id, it = msg
            self.current_task_id = task_id
            self._refresh_history_list()

            # История (левая панель) может быть отключена по НФ — тогда self.lb == None.
            if getattr(self, 'lb', None) is not None:
                self.lb.selection_clear(0, tk.END)
                self.lb.selection_set(0)
                self.lb.activate(0)

            self.status_var.set(f"Задача #{task_id} создана. Запуск оркестрации…")
            self._start_polling()
            return

        if kind == "kick_done":
            _, task_id, _ = msg
            if self.current_task_id == task_id:
                self.status_var.set(f"Задача #{task_id} в работе…")
            return

        if kind == "poll_error":
            _, task_id, detail = msg
            if self.current_task_id == task_id:
                self.status_var.set(f"Ошибка опроса задачи #{task_id}")
                self._append("Ошибка", detail)
            return

        if kind == "show_logs":
            _, task_id, txt = msg
            if self.current_task_id == task_id:
                self.status_var.set(f"Логи задачи #{task_id}")
            self._popup_text(f"Логи задачи #{task_id}", txt)
            return

        if kind == "task_update":
            _, task_id, task = msg
            if self.current_task_id != task_id:
                return

            st = str(task.get("status") or "")
            if st and st != self.last_status:
                self.last_status = st
                self.status_var.set(f"{status_ru(st)}: задача #{task_id}")

            # show question to user if present
            qtxt = extract_question_to_user(task)
            if qtxt and not self._should_hide_aa_line(qtxt):
                self._append_aa(qtxt)

            # show final answer if done
            if str(task.get("status") or "").upper() == "DONE":
                ans = extract_final_answer(task)
                if ans:
                    self._append_aa(ans)
                else:
                    # if DONE but empty — show minimal info
                    self._append_aa("Готово. Финальный ответ пустой (проверь логи задачи).")
                self._polling = False
                return

            if str(task.get("status") or "").upper() in ("FAILED", "CANCELLED"):
                detail = ""
                if isinstance(task.get("result"), dict) and task["result"].get("final_answer"):
                    detail = str(task["result"]["final_answer"])
                elif isinstance(task.get("error"), dict):
                    detail = json.dumps(task["error"], ensure_ascii=False)
                elif isinstance(task.get("error"), str):
                    detail = task["error"]
                self._append("Ошибка", detail or "Задача завершилась ошибкой (см. логи).")
                self._polling = False
                return

            # keep polling
            return

    def on_close(self):
        self._stop = True
        self.destroy()


def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        safe_log(traceback.format_exc())
        raise
