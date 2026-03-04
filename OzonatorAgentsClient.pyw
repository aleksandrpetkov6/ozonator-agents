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
import os
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


DEFAULT_AA_BASE_URL = (os.getenv("OZONATOR_AA_BASE_URL") or "https://ozonator-aa-dev.onrender.com").rstrip("/")
AA_BEARER = (os.getenv("OZONATOR_AA_BEARER") or "").strip()
AA_ADMIN_TOKEN = (os.getenv("OZONATOR_AA_ADMIN_TOKEN") or "").strip()

DEFAULT_LLM_PROVIDER = (os.getenv("OZONATOR_LLM_PROVIDER") or "groq").strip()
DEFAULT_LLM_MODEL = (os.getenv("OZONATOR_LLM_MODEL") or "llama-3.3-70b-versatile").strip()

CREATE_RETRIES = 3
HTTP_TIMEOUT = 45

FAST_POLL_MS = 250
NORMAL_POLL_MS = 1000
FAST_POLL_WINDOW_SEC = 12

POLL_TIMEOUT_SEC = 120

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

    def _request_raw(self, method: str, url: str, body: bytes | None, headers: dict) -> ApiResponse:
        req = urllib_request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
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
            # сеть/SSL/DNS/timeout — покажем причину
            return ApiResponse(False, 0, None, f"URLError: {e}")
        except Exception as e:
            return ApiResponse(False, 0, None, f"{e.__class__.__name__}: {e}")

    def _request_json(self, method: str, url: str, body: dict | None) -> ApiResponse:
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
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
            # сеть/SSL/DNS/timeout — покажем причину
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
        backoff = 0.8
        for attempt in range(CREATE_RETRIES + 1):
            for pref in self._prefixes():
                self.api_prefix = pref
                url = self._mk("/tasks/create")
                resp = self._request_json("POST", url, body)

                if resp.ok and resp.data and resp.data.get("task") and resp.data["task"].get("id"):
                    return int(resp.data["task"]["id"])

                detail = ""
                if isinstance(resp.data, dict):
                    detail = str(resp.data.get("detail") or resp.data.get("message") or "").strip()

                if resp.status:
                    last_err = f"{resp.error or 'create_task_failed'} ({resp.status})"
                else:
                    last_err = resp.error or "create_task_failed"
                if detail:
                    last_err = f"{last_err}: {detail}"

                # На сетевых ошибках/таймаутах — backoff
                if resp.status == 0:
                    import time as _t
                    _t.sleep(backoff)
                    backoff = min(backoff * 1.7, 6.0)

            if attempt < CREATE_RETRIES:
                import time as _t
                _t.sleep(backoff)
                backoff = min(backoff * 1.7, 6.0)

        raise RuntimeError(f"{last_err or 'create_task_failed'} · сервер: {self.base_url}")

    def run_task(self, task_id: int) -> None:
        # если не смогли запустить обработку — лучше сразу сообщить, чем зависнуть
        last_err = None
        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/aa/run-task/{task_id}")
            resp = self._request_json("POST", url, {})
            if resp.ok:
                return
            last_err = resp.error or (resp.data.get("message") if isinstance(resp.data, dict) else None)
        raise RuntimeError(last_err or "run_task_failed")

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

    def upload_file(self, task_id: int, file_path: str) -> None:
        file_name = os.path.basename(file_path) or "file"
        ctype = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        with open(file_path, "rb") as f:
            content = f.read()

        boundary = "----ozonatorboundary" + uuid.uuid4().hex
        head = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + content + tail

        last_err = None
        for pref in self._prefixes():
            self.api_prefix = pref
            url = self._mk(f"/tasks/{task_id}/files/upload")
            headers = self._headers().copy()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            headers["Accept"] = "application/json"
            resp = self._request_raw("POST", url, body, headers)
            if resp.ok:
                return

            detail = ""
            if isinstance(resp.data, dict):
                detail = str(resp.data.get("detail") or resp.data.get("message") or "").strip()

            if resp.status:
                last_err = f"{resp.error or 'upload_failed'} ({resp.status})"
            else:
                last_err = resp.error or "upload_failed"
            if detail:
                last_err = f"{last_err}: {detail}"

        raise RuntimeError(f"{last_err or 'upload_failed'} · сервер: {self.base_url}")


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
        self._attached_files: list[str] = []

        self._build_ui()
        self._bind_hotkeys()

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

    def _append(self, who: str, text: str):
        stamp = datetime.now().strftime("%H:%M")
        self.chat.configure(state="normal")
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert(tk.END, "\n")
        self.chat.insert(tk.END, f"{stamp}\n", ("ts",))
        tag = "user" if who == "Ты" else "aa"
        self.chat.insert(tk.END, f"{who}\n", (tag,))
        self.chat.insert(tk.END, f"{text.strip()}\n")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

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
        if self._attached_files:
            self.lbl_status.config(text=f"● online · файлов: {len(self._attached_files)}")
        else:
            self.lbl_status.config(text="● online")

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
        self._append("Ты", user_text)
        if self._attached_files:
            names = [os.path.basename(p) or "file" for p in self._attached_files]
            self._append("Ты", "Прикреплено файлов: " + ", ".join(names))
        self._push_history("user", user_text)

        self._show_typing()

        payload = self._build_task_payload(user_text)

        def worker():
            try:
                task_id = self.client.create_task(payload)
                self.current_task_id = task_id

                # Загружаем вложения ДО оркестрации
                for p in list(self._attached_files):
                    self.client.upload_file(task_id, p)
                self._clear_files()

                self.client.run_task(task_id)
                self._polling = True
                self._poll_started_at = time.time()
                self._poll_fast_until = time.time() + FAST_POLL_WINDOW_SEC
            except Exception as e:
                self._polling = False
                self.current_task_id = None
                self.after(0, lambda: self._on_error(f"Не удалось отправить задачу: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, msg: str):
        self._clear_typing_if_any()
        messagebox.showerror("Ошибка", msg)

    def _current_poll_interval_ms(self) -> int:
        return FAST_POLL_MS if time.time() < self._poll_fast_until else NORMAL_POLL_MS

    def _tick(self):
        interval = self._current_poll_interval_ms()

        if self._polling and self._poll_started_at and (time.time() - self._poll_started_at) > POLL_TIMEOUT_SEC:
            self._polling = False
            self._clear_typing_if_any()
            self._append(AA_DISPLAY_NAME, "Нет ответа за 2 минуты. Нажми ‘Логи’ — там причина.")
            self._push_history("assistant", "Нет ответа за 2 минуты. Нажми ‘Логи’ — там причина.")

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
        self.lbl_status.config(text=f"● online · задача #{task_id} · {status or 'UNKNOWN'}")
        result = task.get("result") if isinstance(task.get("result"), dict) else {}

        # финальный ответ всегда из task.result.final_answer
        final_answer = ""
        if isinstance(result, dict):
            final_answer = str(result.get("final_answer") or "").strip()

        if final_answer:
            self._polling = False
            self._clear_typing_if_any()
            self._append(AA_DISPLAY_NAME, final_answer)
            self._push_history("assistant", final_answer)
            return

        if status in {"FAILED"}:
            self._polling = False
            self._clear_typing_if_any()
            err = str(task.get("error_message") or "Задача завершилась с ошибкой").strip()
            self._append(AA_DISPLAY_NAME, err)
            self._push_history("assistant", err)

    def _open_logs(self):
        if self.current_task_id is None:
            messagebox.showinfo("Логи", "Сначала отправь задачу, чтобы появился task_id.")
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
