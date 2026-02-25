import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from db.health import check_postgres, check_redis
from db.init_schema import init_schema
from db.inspect import (
    list_agent_instructions,
    list_communication_rules,
    list_public_tables,
)
from db.seed import seed_core_data

app = FastAPI(title="Ozonator Agents AA")


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
