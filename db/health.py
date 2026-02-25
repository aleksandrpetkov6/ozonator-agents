import psycopg
from redis import Redis


def check_postgres(database_url: str | None) -> tuple[bool, str]:
    if not database_url:
        return False, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True, "Postgres OK"
    except Exception as e:
        # Секреты/DSN не выводим
        return False, f"Ошибка Postgres: {e.__class__.__name__}"


def check_redis(redis_url: str | None) -> tuple[bool, str]:
    if not redis_url:
        return False, "REDIS_URL не задан"

    client = None
    try:
        client = Redis.from_url(
            redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        client.ping()
        return True, "Redis OK"
    except Exception as e:
        # Секреты/URL не выводим
        return False, f"Ошибка Redis: {e.__class__.__name__}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
