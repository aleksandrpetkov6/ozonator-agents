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


def list_agent_instructions(database_url: str | None) -> tuple[bool, list[dict] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    sql = """
    SELECT id, agent_code, version, title, is_active, created_at
    FROM agent_instructions
    ORDER BY agent_code, version, id;
    """

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        items = []
        for row in rows:
            items.append(
                {
                    "id": row[0],
                    "agent_code": row[1],
                    "version": row[2],
                    "title": row[3],
                    "is_active": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                }
            )

        return True, items, "OK"
    except Exception as e:
        return False, None, f"Ошибка чтения agent_instructions: {e.__class__.__name__}"
