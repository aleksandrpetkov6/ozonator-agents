from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from db.health import check_postgres, check_redis
from db.init_schema import init_schema
from db.inspect import list_public_tables

app = FastAPI(title="Ozonator Agents AA")


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


@app.post("/admin/init-db")
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


@app.get("/admin/tables")
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
