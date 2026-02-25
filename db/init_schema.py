from pathlib import Path

import psycopg


def init_schema(database_url: str | None) -> tuple[bool, str]:
    if not database_url:
        return False, "DATABASE_URL не задан"

    sql_file = Path(__file__).resolve().parent / "sql" / "001_init.sql"

    if not sql_file.exists():
        return False, f"SQL файл не найден: {sql_file.name}"

    try:
        sql_text = sql_file.read_text(encoding="utf-8")

        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
            conn.commit()

        return True, f"Schema initialized from {sql_file.name}"
    except Exception as e:
        return False, f"Ошибка инициализации схемы: {e.__class__.__name__}"
