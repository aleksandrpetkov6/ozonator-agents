import psycopg


def list_public_tables(database_url: str | None) -> tuple[bool, list[str] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    sql = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
    ORDER BY tablename;
    """

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        tables = [row[0] for row in rows]
        return True, tables, "OK"
    except Exception as e:
        return False, None, f"Ошибка чтения таблиц: {e.__class__.__name__}"
