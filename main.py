import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Annotated, Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from db.health import check_postgres, check_redis
from db.init_schema import init_schema
from db.inspect import (
    list_agent_instructions,
    list_communication_rules,
    list_public_tables,
)
from db.seed import seed_core_data
from db.tasks import (
    create_task_record,
    get_task_logs,
    get_task_record,
    set_task_result,
    update_task_status,
    write_orchestration_log,
    list_recent_user_tasks,
)
from db.files import add_task_file, get_task_file_content, list_task_files
from schemas.tasks import TaskCreateRequest

app = FastAPI(title="Ozonator Agents AA")


def _is_image_file(file_name: str, content_type: str) -> bool:
    ct = (content_type or "").lower().strip()
    if ct.startswith("image/"):
        return True
    ext = (file_name or "").lower().rsplit(".", 1)
    ext = ext[-1] if len(ext) == 2 else ""
    return ext in {"jpg", "jpeg", "png", "webp", "gif"}


def _shrink_image_to_max_bytes(raw: bytes, file_name: str, content_type: str, max_bytes: int) -> tuple[bytes, str]:
    if not raw or len(raw) <= max_bytes:
        return raw, (content_type or "application/octet-stream")

    try:
        from PIL import Image
    except Exception:
        return raw, (content_type or "application/octet-stream")

    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        max_dim = int(os.getenv("AA_UPLOAD_IMAGE_MAX_DIM") or "2600")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        quality = 90

        def encode(q: int) -> bytes:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            return buf.getvalue()

        data = encode(quality)
        while len(data) > max_bytes and quality >= 45:
            quality -= 7
            data = encode(quality)

        attempts = 0
        while len(data) > max_bytes and attempts < 2:
            w, h = img.size
            img = img.resize((max(1, int(w * 0.85)), max(1, int(h * 0.85))))
            data = encode(max(45, quality))
            attempts += 1

        if len(data) <= max_bytes:
            return data, "image/jpeg"
        return raw, (content_type or "application/octet-stream")
    except Exception:
        return raw, (content_type or "application/octet-stream")


def _compact_file_meta(file_row: dict[str, Any] | None) -> dict[str, Any]:
    meta = file_row if isinstance(file_row, dict) else {}
    return {
        "id": int(meta.get("id") or 0),
        "file_name": str(meta.get("file_name") or ""),
        "content_type": str(meta.get("content_type") or "application/octet-stream"),
        "size_bytes": int(meta.get("size_bytes") or 0),
        "created_at": meta.get("created_at"),
    }


def _append_task_result_file_meta(
    database_url: str | None,
    task_id: int,
    *,
    list_key: str,
    file_row: dict[str, Any] | None,
) -> None:
    compact = _compact_file_meta(file_row)
    file_id = int(compact.get("id") or 0)
    if not database_url or not file_id:
        return

    ok_task, task, _msg = get_task_record(database_url, task_id)
    if not ok_task or not isinstance(task, dict):
        return

    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    result = dict(result) if isinstance(result, dict) else {}

    items = result.get(list_key) if isinstance(result.get(list_key), list) else []
    merged: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = int(item.get("id") or 0)
        if not item_id or item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        merged.append(_compact_file_meta(item))

    if file_id not in seen_ids:
        merged.append(compact)

    result[list_key] = merged
    if list_key == "user_upload_files":
        result["user_upload_file_ids"] = [int(x.get("id") or 0) for x in merged if int(x.get("id") or 0)]
    elif list_key == "download_files":
        result["download_file_ids"] = [int(x.get("id") or 0) for x in merged if int(x.get("id") or 0)]

    set_task_result(
        database_url,
        task_id=task_id,
        result=result,
        error_message=task.get("error_message"),
    )


def require_admin_token(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    server_token = os.getenv("ADMIN_DEV_TOKEN")
    if not server_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_DEV_TOKEN не настроен на сервере",
        )
    if not x_admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не указан X-Admin-Token",
        )
    if x_admin_token != server_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный X-Admin-Token",
        )


@app.get("/")
def root():
    return {
        "service": "AA",
        "status": "ok",
        "message": "Агент-администратор запущен",
    }


@app.get("/health")
def health():
    return {
        "service": "AA",
        "status": "ok",
        "message": "Сервис работает",
    }


@app.get("/health/db")
def health_db():
    settings = get_settings()
    ok, message = check_postgres(settings.database_url)
    payload = {
        "service": "AA",
        "component": "postgres",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/health/redis")
def health_redis():
    settings = get_settings()
    ok, message = check_redis(settings.redis_url)
    payload = {
        "service": "AA",
        "component": "redis",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/health/all")
def health_all():
    settings = get_settings()
    db_ok, db_message = check_postgres(settings.database_url)
    redis_ok, redis_message = check_redis(settings.redis_url)
    app_ok = True
    all_ok = app_ok and db_ok and redis_ok
    payload = {
        "service": "AA",
        "status": "ok" if all_ok else "error",
        "components": {
            "app": {
                "status": "ok",
                "message": "Сервис работает",
            },
            "postgres": {
                "status": "ok" if db_ok else "error",
                "message": db_message,
            },
            "redis": {
                "status": "ok" if redis_ok else "error",
                "message": redis_message,
            },
        },
    }
    return JSONResponse(status_code=200 if all_ok else 503, content=payload)


@app.post("/admin/init-db", dependencies=[Depends(require_admin_token)])
def admin_init_db():
    settings = get_settings()
    ok, message = init_schema(settings.database_url)
    payload = {
        "service": "AA",
        "operation": "init_db_schema",
        "status": "ok" if ok else "error",
        "message": message,
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/tables", dependencies=[Depends(require_admin_token)])
def admin_tables():
    settings = get_settings()
    ok, tables, message = list_public_tables(settings.database_url)
    payload = {
        "service": "AA",
        "operation": "list_public_tables",
        "status": "ok" if ok else "error",
        "message": message,
        "tables": tables if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.post("/admin/seed-core", dependencies=[Depends(require_admin_token)])
def admin_seed_core():
    settings = get_settings()
    ok, details, message = seed_core_data(settings.database_url)
    payload = {
        "service": "AA",
        "operation": "seed_core_data",
        "status": "ok" if ok else "error",
        "message": message,
        "details": details if ok else {},
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/agent-instructions", dependencies=[Depends(require_admin_token)])
def admin_agent_instructions():
    settings = get_settings()
    ok, items, message = list_agent_instructions(settings.database_url)
    payload = {
        "service": "AA",
        "operation": "list_agent_instructions",
        "status": "ok" if ok else "error",
        "message": message,
        "items": items if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.get("/admin/communication-rules", dependencies=[Depends(require_admin_token)])
def admin_communication_rules():
    settings = get_settings()
    ok, items, message = list_communication_rules(settings.database_url)
    payload = {
        "service": "AA",
        "operation": "list_communication_rules",
        "status": "ok" if ok else "error",
        "message": message,
        "items": items if ok else [],
    }
    return JSONResponse(status_code=200 if ok else 503, content=payload)


@app.post("/tasks/create")
def tasks_create(body: TaskCreateRequest):
    settings = get_settings()
    ok, task, message = create_task_record(settings.database_url, body.model_dump())
    if not ok and message == "external_task_id уже существует":
        return JSONResponse(
            status_code=409,
            content={
                "service": "AA",
                "operation": "create_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "service": "AA",
            "operation": "create_task",
            "status": "ok" if ok else "error",
            "message": message,
            "task": task if ok else None,
        },
    )
@app.get("/history/recent")
def history_recent(user_key: str | None = None, user_name: str | None = None, limit: int = 30):
    """История для клиента: последние user_task.

    Нужна, чтобы при переустановке/перезапуске клиент мог восстановить чат.
    """
    settings = get_settings()
    ok, items, message = list_recent_user_tasks(
        settings.database_url,
        user_key=user_key,
        user_name=user_name,
        limit=limit,
    )
    if not ok:
        return _json_response(
            503,
            {
                "service": "AA",
                "operation": "history_recent",
                "status": "error",
                "message": message,
                "items": None,
            },
        )

    return _json_response(
        200,
        {
            "service": "AA",
            "operation": "history_recent",
            "status": "ok",
            "message": "OK",
            "items": items or [],
        },
    )


@app.post("/tasks/{task_id}/files/upload")
async def upload_task_file(task_id: int, file: UploadFile = File(...)):
    settings = get_settings()

    ok_task, task, task_message = get_task_record(settings.database_url, task_id)
    if not ok_task or not task:
        return JSONResponse(
            status_code=404,
            content={
                "service": "AA",
                "operation": "upload_task_file",
                "status": "error",
                "message": task_message or "Задача не найдена",
            },
        )

    raw = await file.read()
    original_size = len(raw or b"")
    content_type = file.content_type or "application/octet-stream"
    file_name = file.filename or "file"

    upload_limit = int(os.getenv("AA_UPLOAD_FILE_MAX_BYTES") or "4500000")
    if raw and original_size > upload_limit and _is_image_file(file_name, content_type):
        shrunk, new_ct = _shrink_image_to_max_bytes(raw, file_name, content_type, upload_limit)
        if len(shrunk or b"") < original_size:
            raw = shrunk
            content_type = new_ct
            file_name_root = file_name.rsplit(".", 1)[0]
            if not file_name.lower().endswith(".jpg") and not file_name.lower().endswith(".jpeg"):
                file_name = f"{file_name_root}_server.jpg"

    ok_file, file_row, message = add_task_file(
        settings.database_url,
        task_id=task_id,
        file_name=file_name,
        content_type=content_type,
        content=raw,
    )

    level = "info" if ok_file else "error"
    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="file_uploaded" if ok_file else "file_upload_failed",
        level=level,
        message=(
            "Пользователь загрузил файл во вложения задачи"
            if ok_file
            else "Не удалось сохранить файл во вложения задачи"
        ),
        meta={
            "file_name": file_name,
            "content_type": content_type,
            "original_size_bytes": original_size,
            "stored_size_bytes": len(raw or b""),
            "shrunk": len(raw or b"") < original_size,
            "task_type": task.get("task_type"),
            "user_key": task.get("user_key"),
            "user_name": task.get("user_name"),
            "file": file_row if ok_file else None,
            "error": None if ok_file else message,
        },
    )

    if ok_file and isinstance(file_row, dict):
        _append_task_result_file_meta(
            settings.database_url,
            task_id=task_id,
            list_key="user_upload_files",
            file_row=file_row,
        )

    return JSONResponse(
        status_code=200 if ok_file else 503,
        content={
            "service": "AA",
            "operation": "upload_task_file",
            "status": "ok" if ok_file else "error",
            "message": message,
            "file": file_row if ok_file else None,
        },
    )


@app.get("/tasks/{task_id}/files")
def get_task_files(task_id: int):
    settings = get_settings()

    ok_task, task, task_message = get_task_record(settings.database_url, task_id)
    if not ok_task or not task:
        return JSONResponse(
            status_code=404,
            content={
                "service": "AA",
                "operation": "list_task_files",
                "status": "error",
                "message": task_message or "Задача не найдена",
                "files": [],
            },
        )

    ok_files, files, message = list_task_files(settings.database_url, task_id)
    return JSONResponse(
        status_code=200 if ok_files else 503,
        content={
            "service": "AA",
            "operation": "list_task_files",
            "status": "ok" if ok_files else "error",
            "message": message,
            "files": files if ok_files else [],
        },
    )


@app.get("/tasks/{task_id}/files/{file_id}/download")
def download_task_file(task_id: int, file_id: int):
    settings = get_settings()

    ok_task, task, task_message = get_task_record(settings.database_url, task_id)
    if not ok_task or not task:
        raise HTTPException(status_code=404, detail=task_message or "Задача не найдена")

    ok_file, item, content, message = get_task_file_content(settings.database_url, task_id, file_id)
    if not ok_file or not item:
        raise HTTPException(status_code=404, detail=message or "Файл не найден")

    filename = str(item.get("file_name") or f"file_{file_id}")
    content_type = str(item.get("content_type") or "application/octet-stream")
    safe_ascii_name = ''.join(ch if 32 <= ord(ch) < 127 and ch not in {'"', '\\'} else '_' for ch in filename) or f"file_{file_id}"
    encoded_name = urllib_parse.quote(filename)

    headers = {
        "Content-Disposition": f'attachment; filename="{safe_ascii_name}"; filename*=UTF-8''{encoded_name}'
    }
    return Response(content=content or b"", media_type=content_type, headers=headers)


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    return JSONResponse(
        status_code=200 if ok else 404,
        content={
            "service": "AA",
            "operation": "get_task",
            "status": "ok" if ok else "error",
            "message": message,
            "task": task if ok else None,
        },
    )


@app.get("/tasks/{task_id}/logs")
def task_logs(task_id: int):
    settings = get_settings()
    ok, logs, message = get_task_logs(settings.database_url, task_id)
    return JSONResponse(
        status_code=200 if ok else 404,
        content={
            "service": "AA",
            "operation": "get_task_logs",
            "status": "ok" if ok else "error",
            "message": message,
            "task_id": task_id,
            "count": len(logs) if ok else 0,
            "logs": logs if ok else [],
        },
    )


def _json_response(status_code: int, payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


def _agent_service_url(settings, agent_code: str) -> str:
    if agent_code == "AZ":
        return settings.az_run_task_base_url.rstrip("/")
    if agent_code == "AS":
        return settings.as_run_task_base_url.rstrip("/")
    if agent_code == "AK":
        return settings.ak_run_task_base_url.rstrip("/")
    raise ValueError(f"Неизвестный агент для оркестрации: {agent_code}")


def _agent_run_task_url(settings, agent_code: str, task_id: int) -> str:
    service_url = _agent_service_url(settings, agent_code)
    run_suffix = f"/{agent_code.lower()}/run-task"
    lower_service_url = service_url.lower()

    if lower_service_url.endswith(run_suffix):
        return f"{service_url}/{task_id}"
    if lower_service_url.endswith("/run-task"):
        return f"{service_url}/{task_id}"
    return f"{service_url}{run_suffix}/{task_id}"


def _agent_health_url(settings, agent_code: str) -> str:
    service_url = _agent_service_url(settings, agent_code)
    run_suffix = f"/{agent_code.lower()}/run-task"
    lower_service_url = service_url.lower()

    if lower_service_url.endswith(run_suffix):
        return service_url[: -len(run_suffix)] + "/health"
    if lower_service_url.endswith("/run-task"):
        return service_url.rsplit("/", 1)[0] + "/health"
    return f"{service_url}/health"


def _normalize_execution_response(agent_code: str, payload: dict[str, Any]) -> dict[str, Any]:
    task = payload.get("task") if isinstance(payload.get("task"), dict) else None
    execution_result = payload.get("execution_result") if isinstance(payload.get("execution_result"), dict) else None
    response_status = str(payload.get("status") or "").lower()

    handoff_ready = bool(execution_result and execution_result.get("handoff_ready"))
    next_agent = str(execution_result.get("next_agent") or "").upper() if execution_result else ""
    task_status = str((task or {}).get("status") or (execution_result or {}).get("task_status") or "").upper()

    ok = False
    if response_status == "ok":
        if agent_code == "AZ":
            ok = handoff_ready and next_agent == "AS" and task_status in {"BRIEF_READY", "AZ_DONE"}
        elif agent_code == "AS":
            ok = handoff_ready and next_agent == "AK" and task_status in {"ARTIFACTS_READY", "AS_DONE"}
        elif agent_code == "AK":
            ok = task_status == "DONE"

    return {
        "ok": ok,
        "response": payload,
        "task": task,
        "execution_result": execution_result,
        "task_status": task_status,
        "next_agent": next_agent or None,
        "handoff_ready": handoff_ready,
    }


def _retry_after_seconds(headers: Any) -> int | None:
    if headers is None:
        return None

    raw = None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        raw = None

    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    if text.isdigit():
        return max(1, min(int(text), 60))

    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = int((dt - datetime.now(timezone.utc)).total_seconds())
        return max(1, min(delta, 60))
    except Exception:
        return None


def _warm_agent_service(settings, agent_code: str, timeout: int = 12) -> bool:
    url = _agent_health_url(settings, agent_code)
    req = urllib_request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "ozonator-aa/1.0",
        },
    )
    try:
        with urllib_request.urlopen(req, timeout=max(2, timeout)) as resp:
            try:
                resp.read(256)
            except Exception:
                pass
            return int(getattr(resp, "status", 200) or 200) < 500
    except Exception:
        return False


def _retry_sleep_seconds(http_status: int | None, attempt: int, delay: float, retry_after: int | None = None) -> float:
    if retry_after is not None:
        return float(max(1, min(retry_after, 60)))

    base = max(0.5, float(delay))
    if http_status == 429:
        return min(max(base, 8.0) * max(1, attempt), 60.0)
    if http_status in {502, 503, 504}:
        return min(max(base, 4.0) * max(1, attempt), 45.0)
    return min(max(base, 2.0), 15.0)


def _build_step_meta(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    execution_result = response.get("execution_result") if isinstance(response.get("execution_result"), dict) else {}
    task = response.get("task") if isinstance(response.get("task"), dict) else {}

    return {
        "agent": payload.get("agent"),
        "url": payload.get("url"),
        "http_status": payload.get("http_status"),
        "attempts": payload.get("attempts"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "retry_after_sec": payload.get("retry_after_sec"),
        "next_sleep_sec": payload.get("next_sleep_sec"),
        "warmup_attempted": payload.get("warmup_attempted"),
        "warmup_ok": payload.get("warmup_ok"),
        "transport_error": payload.get("error"),
        "response_status": response.get("status"),
        "response_message": response.get("message"),
        "task_status": task.get("status"),
        "next_agent": execution_result.get("next_agent"),
        "response": response if response else None,
    }


def _log_agent_call(settings, task_id: int, cycle_no: int, agent_code: str, ok: bool, payload: dict[str, Any]) -> None:
    meta = {"cycle_no": cycle_no, **_build_step_meta(payload)}
    level = "info" if ok else "error"
    if ok and meta.get("response_status") not in {None, "ok"}:
        level = "warning"

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="aa_agent_call_completed",
        level=level,
        message=f"AA вызвал {agent_code.upper()}",
        meta=meta,
    )


def _call_agent(settings, task_id: int, agent_code: str, max_retries: int = 8, initial_delay_sec: float = 4.0) -> tuple[bool, dict[str, Any]]:
    url = _agent_run_task_url(settings, agent_code, task_id)
    delay = max(1.0, initial_delay_sec)
    last_payload: dict[str, Any] | None = None
    started_at = time.time()
    total_wait_limit = 180.0

    for attempt in range(1, max_retries + 1):
        retry_after: int | None = None
        http_status: int | None = None
        should_warm_before_retry = False
        warmed = False

        try:
            req = urllib_request.Request(
                url,
                method="POST",
                data=b"",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "ozonator-aa/1.0",
                },
            )
            with urllib_request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                normalized = _normalize_execution_response(agent_code, parsed)
                payload = {
                    "agent": agent_code,
                    "url": url,
                    "attempts": attempt,
                    "elapsed_sec": round(time.time() - started_at, 2),
                    "http_status": getattr(resp, "status", 200),
                    "response": parsed,
                    **normalized,
                }
                if normalized["ok"]:
                    return True, payload
                last_payload = payload
                should_warm_before_retry = True
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            parsed: dict[str, Any] | None = None
            if body:
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = {"raw": body}
            http_status = getattr(exc, "code", None)
            retry_after = _retry_after_seconds(getattr(exc, "headers", None))
            should_warm_before_retry = http_status in {502, 503, 504}
            payload = {
                "agent": agent_code,
                "url": url,
                "attempts": attempt,
                "elapsed_sec": round(time.time() - started_at, 2),
                "http_status": http_status,
                "retry_after_sec": retry_after,
                "error": f"HTTPError: {exc}",
                "response": parsed,
                **(_normalize_execution_response(agent_code, parsed) if isinstance(parsed, dict) else {
                    "ok": False,
                    "task": None,
                    "execution_result": None,
                    "task_status": None,
                    "next_agent": None,
                    "handoff_ready": False,
                }),
            }
            last_payload = payload
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            should_warm_before_retry = True
            last_payload = {
                "agent": agent_code,
                "url": url,
                "attempts": attempt,
                "elapsed_sec": round(time.time() - started_at, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
                "response": None,
                "task": None,
                "execution_result": None,
                "task_status": None,
                "next_agent": None,
                "handoff_ready": False,
            }

        if attempt < max_retries:
            sleep_for = _retry_sleep_seconds(http_status, attempt, delay, retry_after)
            elapsed = time.time() - started_at
            remaining = total_wait_limit - elapsed
            if remaining <= 0:
                break
            sleep_for = min(sleep_for, max(1.0, remaining))
            if should_warm_before_retry:
                warmed = _warm_agent_service(settings, agent_code, timeout=min(int(max(5.0, sleep_for)), 20))
            if isinstance(last_payload, dict):
                last_payload["warmup_attempted"] = bool(should_warm_before_retry)
                last_payload["warmup_ok"] = bool(warmed)
                last_payload["next_sleep_sec"] = round(float(sleep_for), 2)
            time.sleep(sleep_for)
            delay = min(max(delay * 1.7, sleep_for), 60.0)

    return False, last_payload or {
        "agent": agent_code,
        "url": url,
        "attempts": 0,
        "elapsed_sec": round(time.time() - started_at, 2),
        "error": "Неизвестная ошибка вызова агента",
        "ok": False,
        "response": None,
        "task": None,
        "execution_result": None,
        "task_status": None,
        "next_agent": None,
        "handoff_ready": False,
    }


def _run_single_cycle(settings, task_id: int, cycle_no: int) -> tuple[bool, dict[str, Any]]:
    step_results: list[dict[str, Any]] = []

    ok_az, az_payload = _call_agent(settings, task_id, "AZ")
    step_results.append(az_payload)
    _log_agent_call(settings, task_id, cycle_no, "AZ", ok_az, az_payload)
    if not ok_az:
        return False, {
            "failed_at": "AZ",
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AZ",
        }

    ok_as, as_payload = _call_agent(settings, task_id, "AS")
    step_results.append(as_payload)
    _log_agent_call(settings, task_id, cycle_no, "AS", ok_as, as_payload)
    if not ok_as:
        return False, {
            "failed_at": "AS",
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AS",
        }

    ok_ak, ak_payload = _call_agent(settings, task_id, "AK")
    step_results.append(ak_payload)
    _log_agent_call(settings, task_id, cycle_no, "AK", ok_ak, ak_payload)
    if not ok_ak:
        return False, {
            "failed_at": "AK",
            "steps": step_results,
            "message": "AA не смог автоматически вызвать AK",
        }

    final_task = ak_payload.get("task") or {}
    final_status = str(final_task.get("status") or "").upper()
    if final_status != "DONE":
        return False, {
            "failed_at": "AK",
            "steps": step_results,
            "message": "AA завершил цепочку, но финальный статус задачи не DONE",
        }

    return True, {
        "failed_at": None,
        "steps": step_results,
        "message": "Цепочка AA → AZ → AS → AK завершена",
    }


def _run_auto_orchestration(settings, task_id: int, max_cycles: int = 2) -> dict[str, Any]:
    cycles: list[dict[str, Any]] = []

    for cycle_no in range(1, max_cycles + 1):
        ok, cycle_result = _run_single_cycle(settings, task_id, cycle_no)
        cycle_payload = {
            "cycle_no": cycle_no,
            **cycle_result,
        }
        cycles.append(cycle_payload)

        if ok:
            return {
                "orchestration_status": "done",
                "message": "AA завершил автоматическую оркестрацию цепочки",
                "cycles": cycles,
                "failed_at": None,
            }

        failed_at = str(cycle_result.get("failed_at") or "").upper()
        if failed_at == "AZ":
            break

        time.sleep(0.5)

    failed_at = str(cycles[-1].get("failed_at") or "").upper() if cycles else "UNKNOWN"
    return {
        "orchestration_status": "failed",
        "message": cycles[-1].get("message") if cycles else "AA не смог завершить автоматическую оркестрацию",
        "cycles": cycles,
        "failed_at": failed_at,
    }


def _finalize_orchestration_result(settings, task_id: int, orchestration_result: dict[str, Any]) -> None:
    # Не даём задаче зависнуть без финального ответа.
    status_raw = str(orchestration_result.get("orchestration_status") or "").lower()
    if status_raw == "done":
        return

    ok, task, _msg = get_task_record(settings.database_url, task_id)
    if not ok or not isinstance(task, dict):
        return

    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    current_result = current_result if isinstance(current_result, dict) else {}

    existing_final = str(current_result.get("final_answer") or "").strip()
    if existing_final:
        return

    failed_at = str(orchestration_result.get("failed_at") or "").upper() or None
    preserve_az_handoff = (
        status_raw == "failed"
        and failed_at == "AZ"
        and (
            str(current_result.get("next_agent") or "").upper() == "AZ"
            or str(current_result.get("routed_to") or "").upper() == "AZ"
        )
    )

    if status_raw == "review_needs_attention":
        user_msg = (
            "Задача дошла до проверки, но нужна доработка по замечаниям. "
            "Нажми ‘Логи’ в клиенте — там подробности и следующий шаг."
        )
        new_status = "REVIEW_NEEDS_ATTENTION"
    elif status_raw == "failed":
        if preserve_az_handoff:
            user_msg = (
                "Задача не завершилась автоматически на шаге AZ. "
                "Handoff в AZ сохранён: нажми ‘Логи’ и при необходимости повтори запуск AZ."
            )
            new_status = "AA_ROUTED"
        else:
            user_msg = (
                f"Задача не завершилась: не удалось выполнить следующий шаг ({failed_at or 'неизвестно'}). "
                "Нажми ‘Логи’ в клиенте — там причина и детали."
            )
            new_status = "FAILED"
    else:
        user_msg = (
            "Задача не завершилась автоматически (неожиданный финальный статус). "
            "Нажми ‘Логи’ в клиенте — там детали."
        )
        new_status = "FAILED"

    merged = {
        **current_result,
        "final_answer": user_msg,
        "aa_orchestration": orchestration_result,
    }

    if preserve_az_handoff:
        merged.update(
            {
                "handoff_ready": True,
                "next_agent": current_result.get("next_agent") or "AZ",
                "routed_to": current_result.get("routed_to") or "AZ",
                "aa_retry_available": "AZ",
            }
        )
    else:
        merged.update(
            {
                "handoff_ready": False,
                "next_agent": None,
            }
        )

    set_task_result(
        settings.database_url,
        task_id=task_id,
        result=merged,
        error_message=str(orchestration_result.get("message") or "").strip() or None,
    )
    update_task_status(settings.database_url, task_id, new_status)


@app.post("/aa/run-task/{task_id}")
def aa_run_task(task_id: int):
    settings = get_settings()

    ok_task, task, task_message = get_task_record(settings.database_url, task_id)
    if not ok_task or not task:
        return _json_response(
            404,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": task_message or "Задача не найдена",
                "task": None,
                "execution_result": None,
            },
        )

    current_status = str(task.get("status") or "").upper()
    if current_status not in {"NEW", "IN_PROGRESS", "FAILED", "AA_ROUTED", "AZ_DONE", "AS_DONE"}:
        return _json_response(
            400,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": (
                    "AA может принимать задачу только в статусах NEW/IN_PROGRESS/FAILED/AA_ROUTED/AZ_DONE/AS_DONE"
                ),
                "task": task,
                "execution_result": None,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_run_started",
        level="info",
        message="AA начал обработку задачи",
        meta={
            "task_type": task.get("task_type"),
            "user_key": task.get("user_key"),
            "user_name": task.get("user_name"),
            "current_status": current_status,
        },
    )

    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    attachments: list[dict[str, Any]] = []
    ok_files, files, files_message = list_task_files(settings.database_url, task_id)
    if ok_files:
        attachments = files or []
    else:
        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AA",
            event_type="task_files_unavailable",
            level="warning",
            message="AA не смог получить список файлов задачи перед handoff",
            meta={"error": files_message},
        )

    handoff = {
        "mode": "aa_handoff_v1",
        "note": "AA подготовил задачу и передал handoff в AZ",
        "attachments": [
            {
                "file_id": item.get("id"),
                "file_name": item.get("file_name"),
                "content_type": item.get("content_type"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in attachments
        ],
    }

    current_result = task.get("result") if isinstance(task.get("result"), dict) else {}
    current_result = dict(current_result) if isinstance(current_result, dict) else {}

    execution_result = {
        **current_result,
        "source_agent": "AA",
        "target_agent": "AA",
        "task_type": task.get("task_type"),
        "status": "routed_to_az",
        "next_agent": "AZ",
        "aa_executor": "AA",
        "handoff_ready": True,
        "handoff": handoff,
    }
    set_task_result(
        settings.database_url,
        task_id=task_id,
        result=execution_result,
        error_message=None,
    )
    update_task_status(settings.database_url, task_id, "AA_ROUTED")

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_routed_to_az",
        level="info",
        message="AA подготовил handoff и маршрутизировал задачу в AZ",
        meta={
            "mode": handoff.get("mode"),
            "attachments_count": len(attachments),
            "next_agent": "AZ",
            "handoff_ready": True,
        },
    )

    if not settings.aa_auto_orchestration_enabled:
        disabled_msg = (
            "Автоматическая оркестрация отключена на сервере (AA_AUTO_ORCHESTRATION_ENABLED=false). "
            "Handoff в AZ сохранён: задачу можно запустить вручную."
        )
        merged = {
            **execution_result,
            "final_answer": disabled_msg,
            "handoff_ready": True,
            "next_agent": "AZ",
            "routed_to": "AZ",
            "aa_retry_available": "AZ",
        }
        set_task_result(settings.database_url, task_id=task_id, result=merged, error_message=None)
        write_orchestration_log(
            settings.database_url,
            task_id=task_id,
            actor_agent="AA",
            event_type="aa_auto_orchestration_skipped",
            level="warning",
            message="AA пропустил автоматическую оркестрацию: она отключена на сервисе",
            meta={"handoff_preserved": True, "next_agent": "AZ"},
        )
        return _json_response(
            200,
            {
                "service": "AA",
                "operation": "aa_run_task",
                "status": "ok",
                "message": disabled_msg,
                "task": {
                    **task,
                    "status": "AA_ROUTED",
                    "result": merged,
                },
                "execution_result": merged,
            },
        )

    orchestration_result = _run_auto_orchestration(settings, task_id)

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="aa_auto_orchestration_completed",
        level="info" if orchestration_result.get("orchestration_status") == "done" else "warning",
        message=orchestration_result.get("message") or "AA завершил автоматическую оркестрацию",
        meta={
            "orchestration_status": orchestration_result.get("orchestration_status"),
            "failed_at": orchestration_result.get("failed_at"),
            "message": orchestration_result.get("message"),
            "cycles_count": len(orchestration_result.get("cycles") or []),
            "cycles": orchestration_result.get("cycles") or [],
        },
    )

    _finalize_orchestration_result(settings, task_id, orchestration_result)

    ok_final, final_task, _final_message = get_task_record(settings.database_url, task_id)
    return _json_response(
        200,
        {
            "service": "AA",
            "operation": "aa_run_task",
            "status": "ok" if ok_final else "error",
            "message": orchestration_result.get("message") or "OK",
            "task": final_task if ok_final else None,
            "execution_result": {
                **execution_result,
                "aa_orchestration": orchestration_result,
            },
        },
    )


@app.post("/debug/fill-task/{task_id}", dependencies=[Depends(require_admin_token)])
def debug_fill_task(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok or not task:
        return JSONResponse(
            status_code=404,
            content={
                "service": "AA",
                "operation": "debug_fill_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    task_type = str(task.get("task_type") or "")
    allowed_task_types = {"user_task", "project_task", "review_task", "system_task"}
    if task_type not in allowed_task_types:
        return JSONResponse(
            status_code=400,
            content={
                "service": "AA",
                "operation": "debug_fill_task",
                "status": "error",
                "message": f"debug_fill_task поддерживает только task_type из {sorted(allowed_task_types)}",
                "task": task,
            },
        )

    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    user_name = payload.get("user_name") or task.get("user_name") or "Пользователь"
    prompt = payload.get("prompt") or task.get("content") or ""

    base_preview = (
        f"Привет, {user_name}!\n\n"
        "Это тестовый ответ DEBUG режима AA.\n"
        "Я заполнила финальный результат задачи без вызова внешних агентов.\n\n"
        f"Кратко по запросу: {str(prompt)[:500]}"
    ).strip()

    synthetic_result = {
        "source_agent": "AA",
        "target_agent": "AA",
        "status": "filled_by_debug",
        "debug_mode": True,
        "task_type": task_type,
        "final_answer": base_preview,
        "draft_result": {
            "summary": str(prompt)[:300],
            "note": "Результат сгенерирован debug endpoint /debug/fill-task/{task_id}",
        },
    }

    set_task_result(
        settings.database_url,
        task_id=task_id,
        result=synthetic_result,
        error_message=None,
    )
    update_task_status(settings.database_url, task_id, "DONE")

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="debug_fill_task",
        level="warning",
        message="DEBUG endpoint заполнил финальный результат задачи без запуска оркестрации",
        meta={
            "task_type": task_type,
            "task_status": "DONE",
            "debug_mode": True,
        },
    )

    ok_final, final_task, final_message = get_task_record(settings.database_url, task_id)
    return JSONResponse(
        status_code=200 if ok_final else 503,
        content={
            "service": "AA",
            "operation": "debug_fill_task",
            "status": "ok" if ok_final else "error",
            "message": final_message,
            "task": final_task if ok_final else None,
        },
    )


@app.post("/debug/test-outbound/{agent_code}", dependencies=[Depends(require_admin_token)])
def debug_test_outbound(agent_code: str):
    settings = get_settings()
    code = str(agent_code or "").upper().strip()
    if code not in {"AZ", "AS", "AK"}:
        return _json_response(
            400,
            {
                "service": "AA",
                "operation": "debug_test_outbound",
                "status": "error",
                "message": "Поддерживаются только agent_code: AZ, AS, AK",
                "agent": code,
            },
        )

    health_url = _agent_health_url(settings, code)

    try:
        req = urllib_request.Request(
            health_url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "ozonator-aa/1.0",
            },
        )
        with urllib_request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            return _json_response(
                200,
                {
                    "service": "AA",
                    "operation": "debug_test_outbound",
                    "status": "ok",
                    "message": "AA успешно достучался до внешнего агента",
                    "agent": code,
                    "health_url": health_url,
                    "http_status": getattr(resp, "status", 200),
                    "response": parsed,
                },
            )
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        return _json_response(
            502,
            {
                "service": "AA",
                "operation": "debug_test_outbound",
                "status": "error",
                "message": "Внешний агент ответил ошибкой HTTP",
                "agent": code,
                "health_url": health_url,
                "http_status": getattr(exc, "code", None),
                "body": body,
            },
        )
    except Exception as exc:
        return _json_response(
            502,
            {
                "service": "AA",
                "operation": "debug_test_outbound",
                "status": "error",
                "message": "AA не смог достучаться до внешнего агента",
                "agent": code,
                "health_url": health_url,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )


@app.get("/debug/task-status/{task_id}", dependencies=[Depends(require_admin_token)])
def debug_task_status(task_id: int):
    settings = get_settings()
    ok, task, message = get_task_record(settings.database_url, task_id)
    return JSONResponse(
        status_code=200 if ok else 404,
        content={
            "service": "AA",
            "operation": "debug_task_status",
            "status": "ok" if ok else "error",
            "message": message,
            "task": task if ok else None,
        },
    )
