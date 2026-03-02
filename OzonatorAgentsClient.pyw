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
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error

import tkinter as tk
from tkinter import ttk, messagebox


# =========================
# Config (hidden)
# =========================
DEFAULT_BASE_URL = os.environ.get("OZONATOR_AA_BASE_URL", "https://ozonator-aa-dev.onrender.com").rstrip("/")
AA_BEARER = os.environ.get("OZONATOR_AA_BEARER", "").strip()
AA_ADMIN_TOKEN = os.environ.get("OZONATOR_AA_ADMIN_TOKEN", "").strip()

DEFAULT_LLM_PROVIDER = os.environ.get("OZONATOR_LLM_PROVIDER", "groq").strip() or "groq"
DEFAULT_LLM_MODEL = os.environ.get("OZONATOR_LLM_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"

APP_NAME = "Ozonator Agents Client"
HISTORY_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OzonatorAgentsClient"
HISTORY_FILE = HISTORY_DIR / "history.json"
LOG_FILE = HISTORY_DIR / "client.log"

MAX_CONTEXT_MESSAGES = 20
MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_ITEM_CHARS = 1200

POLL_INTERVAL_SEC = 1.0
HTTP_TIMEOUT_SEC = 60
RUN_TASK_TIMEOUT_SEC = 10  # короткий "пинок" оркестрации — дальше работаем polling'ом
CREATE_RETRIES = 4


def _app_install_dir() -> Path:
    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
    except Exception:
        pass
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _pin_working_directory() -> None:
    try:
        os.chdir(_app_install_dir())
    except Exception:
        pass


_pin_working_directory()


# =========================
# Helpers
# =========================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


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

    def create_task(self, user_text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> int:
        history_payload: List[Dict[str, str]] = []
        for item in (conversation_history or []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            history_payload.append({"role": role, "content": content[:MAX_CONTEXT_ITEM_CHARS]})
        if len(history_payload) > MAX_CONTEXT_MESSAGES:
            history_payload = history_payload[-MAX_CONTEXT_MESSAGES:]

        body = {
            "target_agent": "AA",
            "task_type": "user_task",
            "priority": 100,
            "payload": {
                "user_request": user_text,
                "llm_provider": DEFAULT_LLM_PROVIDER,
                "llm_model": DEFAULT_LLM_MODEL,
                "client_meta": {"source": "desktop", "client": "OzonatorAgentsClient"},
                "conversation_enabled": True,
                "conversation_history": history_payload,
                "conversation_turns": len(history_payload),
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
        self.geometry("1180x720")
        self.minsize(980, 620)

        self.q = queue.Queue()
        self.client = AAClient(DEFAULT_BASE_URL)

        self.items: List[TaskItem] = []
        self.current_task_id: Optional[int] = None
        self.last_status: Optional[str] = None
        self._polling = False
        self._poll_inflight = False
        self._poll_generation = 0
        self._shown_messages = set()
        self.dialog_messages: List[Dict[str, str]] = []
        self._conversation_task_ids = set()
        self._assistant_turn_keys = set()
        self._stop = False

        self._load_history()
        self._build_ui()
        self._refresh_history_list()
        self.after(30, self._maximize_window)

        self.after(200, self._drain_queue)
        self.after(int(POLL_INTERVAL_SEC * 1000), self._tick)

    def _build_ui(self):
        # top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 12, "bold")).pack(side="left")

        ttk.Button(top, text="Логи", command=self._show_logs).pack(side="right")

        # body
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # left: history
        left = ttk.Frame(body, width=240)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="История", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        self.lb = tk.Listbox(left, height=25)
        self.lb.pack(fill="both", expand=True)
        self.lb.bind("<<ListboxSelect>>", self._on_select_history)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Удалить", command=self._delete_selected).pack(side="left")
        ttk.Button(btns, text="Очистить", command=self._clear_history).pack(side="right")

        # right: chat
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.status_var = tk.StringVar(value="Готово к работе. Введите задачу и нажмите «Отправить».")
        ttk.Label(right, textvariable=self.status_var).pack(anchor="w", pady=(0, 6))

        self.chat = tk.Text(right, wrap="word", height=20)
        self.chat.pack(fill="both", expand=True)
        self.chat.configure(state="disabled")

        # input
        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=(8, 0))

        self.input = tk.Text(bottom, wrap="word", height=5, undo=True, autoseparators=True, maxundo=-1)
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_input_return)
        self.input.bind("<KP_Enter>", self._on_input_return)
        self.input.bind("<Control-KeyPress>", self._on_input_ctrl_key)
        self.input.bind("<Control-Insert>", self._on_copy_only)
        self.input.bind("<Shift-Insert>", self._on_paste_only)
        self.input.bind("<Shift-Delete>", self._on_cut_only)
        self.input.bind("<F5>", self._on_insert_datetime)
        self.input.bind("<Button-3>", self._show_input_context_menu)
        self._build_input_context_menu()

        actions = ttk.Frame(bottom)
        actions.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Отправить", command=self._send).pack(fill="x")
        ttk.Button(actions, text="Очистить", command=self._clear_input).pack(fill="x", pady=(6, 0))

    def _maximize_window(self):
        try:
            self.state("zoomed")
            return
        except Exception:
            pass
        try:
            self.attributes("-zoomed", True)
            return
        except Exception:
            pass
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            self.geometry(f"{sw}x{sh}+0+0")
        except Exception:
            pass

    def _build_input_context_menu(self):
        self.input_menu = tk.Menu(self, tearoff=0)
        self.input_menu.add_command(label="Отменить", command=self._input_undo)
        self.input_menu.add_command(label="Повторить", command=self._input_redo)
        self.input_menu.add_separator()
        self.input_menu.add_command(label="Вырезать", command=self._input_cut)
        self.input_menu.add_command(label="Копировать", command=self._input_copy)
        self.input_menu.add_command(label="Вставить", command=self._input_paste)
        self.input_menu.add_command(label="Удалить", command=self._input_delete_selection)
        self.input_menu.add_separator()
        self.input_menu.add_command(label="Выделить всё", command=self._input_select_all)
        self.input_menu.add_command(label="Дата/время", command=self._insert_datetime)

    def _show_input_context_menu(self, event):
        try:
            self.input.focus_set()
            self.input_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self.input_menu.grab_release()
            except Exception:
                pass
        return "break"

    def _input_undo(self):
        try:
            self.input.edit_undo()
        except Exception:
            pass

    def _input_redo(self):
        try:
            self.input.edit_redo()
        except Exception:
            pass

    def _input_copy(self):
        try:
            self.input.event_generate("<<Copy>>")
        except Exception:
            pass

    def _input_cut(self):
        try:
            self.input.event_generate("<<Cut>>")
        except Exception:
            pass

    def _input_paste(self):
        try:
            self.input.event_generate("<<Paste>>")
        except Exception:
            pass

    def _input_delete_selection(self):
        try:
            if self.input.tag_ranges("sel"):
                self.input.delete("sel.first", "sel.last")
        except Exception:
            pass

    def _input_select_all(self):
        try:
            self.input.focus_set()
            self.input.tag_add("sel", "1.0", "end-1c")
            self.input.mark_set("insert", "1.0")
            self.input.see("insert")
        except Exception:
            pass

    def _insert_datetime(self):
        try:
            self.input.insert(tk.INSERT, datetime.now().strftime("%H:%M %d.%m.%Y"))
        except Exception:
            pass

    def _on_insert_datetime(self, _event=None):
        self._insert_datetime()
        return "break"

    def _on_copy_only(self, _event=None):
        self._input_copy()
        return "break"

    def _on_paste_only(self, _event=None):
        self._input_paste()
        return "break"

    def _on_cut_only(self, _event=None):
        self._input_cut()
        return "break"

    def _ctrl_action_from_event(self, event) -> str:
        keycode_map = {
            65: "select_all",
            67: "copy",
            86: "paste",
            88: "cut",
            89: "redo",
            90: "undo",
        }
        action = keycode_map.get(int(getattr(event, "keycode", 0) or 0))
        if action:
            return action
        keysym = str(getattr(event, "keysym", "") or "").lower()
        char = str(getattr(event, "char", "") or "").lower()
        alias = {
            "a": "select_all", "c": "copy", "v": "paste", "x": "cut", "y": "redo", "z": "undo",
            "ф": "select_all", "с": "copy", "м": "paste", "ч": "cut", "н": "redo", "я": "undo",
        }
        return alias.get(char) or alias.get(keysym) or ""

    def _on_input_ctrl_key(self, event):
        action = self._ctrl_action_from_event(event)
        if not action:
            return None
        if action == "select_all":
            self._input_select_all()
        elif action == "copy":
            self._input_copy()
        elif action == "paste":
            self._input_paste()
        elif action == "cut":
            self._input_cut()
        elif action == "undo":
            self._input_undo()
        elif action == "redo":
            self._input_redo()
        return "break"

    def _normalize_msg_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _append_unique(self, scope: str, who: str, text: str):
        norm = self._normalize_msg_text(text)
        if not norm:
            return
        key = (int(self.current_task_id or 0), scope, norm)
        if key in self._shown_messages:
            return
        self._shown_messages.add(key)
        self._append(who, text)

    def _clip_context_text(self, text: str, limit: int = MAX_CONTEXT_ITEM_CHARS) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        return clean[:limit]

    def _build_conversation_history(self, limit: int = MAX_CONTEXT_MESSAGES, max_chars: int = MAX_CONTEXT_CHARS) -> List[Dict[str, str]]:
        prepared: List[Dict[str, str]] = []
        total = 0
        for item in reversed(self.dialog_messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            content = self._clip_context_text(str(item.get("content") or ""))
            if not content:
                continue
            projected = total + len(content)
            if prepared and projected > max_chars:
                break
            prepared.append({"role": role, "content": content})
            total = projected
            if len(prepared) >= limit:
                break
        prepared.reverse()
        return prepared

    def _remember_user_turn(self, text: str):
        content = self._clip_context_text(text)
        if not content:
            return
        self.dialog_messages.append({"role": "user", "content": content})

    def _remember_assistant_turn(self, task_id: int, scope: str, text: str):
        content = self._clip_context_text(text)
        if not content:
            return
        key = (int(task_id or 0), str(scope), content)
        if key in self._assistant_turn_keys:
            return
        self._assistant_turn_keys.add(key)
        if int(task_id or 0) not in self._conversation_task_ids:
            return
        self.dialog_messages.append({"role": "assistant", "content": content})

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
        self.lb.delete(0, tk.END)
        for it in self.items:
            self.lb.insert(tk.END, it.title)

    def _on_select_history(self, _evt=None):
        sel = self.lb.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.items):
            return
        it = self.items[idx]
        self.current_task_id = it.id
        self.client.base_url = it.base_url
        self.client.api_prefix = it.api_prefix or ""
        self._append_system("Выбрана задача.")
        self._start_polling()

    def _delete_selected(self):
        sel = self.lb.curselection()
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
        self._append_system("Удалено из истории.")

    def _clear_history(self):
        if not self.items:
            return
        if messagebox.askyesno("Подтверждение", "Очистить всю историю?"):
            self.items = []
            self.dialog_messages = []
            self._conversation_task_ids.clear()
            self._assistant_turn_keys.clear()
            self._shown_messages.clear()
            self._save_history()
            self._refresh_history_list()
            self.current_task_id = None
            self._append_system("История очищена. Контекст диалога сброшен.")

    # ---------- chat ----------
    def _append(self, who: str, text: str):
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"[{_ts()}] {who}: {text}\n")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

    def _append_system(self, text: str):
        self._append("Система", text)

    def _append_user(self, text: str):
        self._append("Вы", text)

    def _append_aa(self, text: str):
        self._append("AA", text)

    # ---------- actions ----------
    def _clear_input(self):
        self.input.delete("1.0", tk.END)

    def _on_input_return(self, event):
        if event.state & 0x0001:
            self.input.insert(tk.INSERT, "\n")
        else:
            self._send()
        return "break"

    def _send(self):
        user_text = self.input.get("1.0", tk.END).strip()
        if not user_text:
            return
        history_snapshot = self._build_conversation_history()
        self._clear_input()
        self._append_user(user_text)
        self.status_var.set("Отправка задачи…")

        def worker():
            try:
                task_id = self.client.create_task(user_text, history_snapshot)
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

                self.q.put(("task_created", task_id, it, user_text))
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
        self._poll_inflight = False
        self._poll_generation += 1
        self.last_status = None

    def _tick(self):
        if self._stop:
            return
        try:
            if self._polling and self.current_task_id is not None and not self._poll_inflight:
                self._poll_inflight = True
                poll_generation = self._poll_generation

                def poll_worker(task_id: int, generation: int):
                    try:
                        task = self.client.get_task(task_id)
                        self.q.put(("task_update", generation, task_id, task))
                    except Exception as e:
                        self.q.put(("poll_error", generation, task_id, str(e)))
                    finally:
                        self.q.put(("poll_finished", generation, task_id, None))

                threading.Thread(target=poll_worker, args=(self.current_task_id, poll_generation), daemon=True).start()
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
        self.status_var.set("Загрузка логов…")

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
        win.geometry("900x600")
        t = tk.Text(win, wrap="word")
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
            _, task_id, it, user_text = msg
            self._conversation_task_ids.add(int(task_id))
            self._remember_user_turn(user_text)
            self.current_task_id = task_id
            self._refresh_history_list()
            self.lb.selection_clear(0, tk.END)
            self.lb.selection_set(0)
            self.lb.activate(0)
            self.status_var.set("Задача создана. Запуск оркестрации…")
            self._start_polling()
            return

        if kind == "kick_done":
            _, task_id, _ = msg
            if self.current_task_id == task_id:
                self.status_var.set("Задача в работе…")
            return

        if kind == "poll_error":
            _, generation, task_id, detail = msg
            if generation != self._poll_generation:
                return
            if self.current_task_id == task_id:
                self.status_var.set("Ошибка опроса задачи")
                self._append("Ошибка", detail)
            return

        if kind == "poll_finished":
            _, generation, _task_id, _ = msg
            if generation == self._poll_generation:
                self._poll_inflight = False
            return

        if kind == "show_logs":
            _, task_id, txt = msg
            if self.current_task_id == task_id:
                self.status_var.set("Логи задачи")
            self._popup_text("Логи задачи", txt)
            return

        if kind == "task_update":
            _, generation, task_id, task = msg
            if generation != self._poll_generation:
                return
            if self.current_task_id != task_id:
                return

            st = str(task.get("status") or "")
            if st and st != self.last_status:
                self.last_status = st
                self.status_var.set(status_ru(st))

            st_upper = str(task.get("status") or "").upper()

            # Show a blocking question only when task explicitly returned to user for clarification.
            qtxt = extract_question_to_user(task)
            if qtxt and st_upper == "REVIEW_NEEDS_ATTENTION":
                self._append_unique("question", "AA", qtxt)
                self._remember_assistant_turn(task_id, "question", qtxt)

            # show final answer if done
            if st_upper == "DONE":
                ans = extract_final_answer(task)
                if ans:
                    self._append_unique("final", "AA", ans)
                    self._remember_assistant_turn(task_id, "final", ans)
                else:
                    # if DONE but empty — show minimal info
                    self._append_unique("final_empty", "AA", "Готово. Финальный ответ пустой (проверь логи задачи).")
                self._polling = False
                return

            if st_upper in ("FAILED", "CANCELLED"):
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
