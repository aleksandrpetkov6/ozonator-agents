import hashlib
from typing import Any

import psycopg


def _row_to_file(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "task_id": row[1],
        "file_name": row[2],
        "content_type": row[3],
        "size_bytes": int(row[4] or 0),
        "sha256": row[5],
        "created_at": row[6].isoformat() if row[6] else None,
    }


def add_task_file(
    database_url: str | None,
    task_id: int,
    file_name: str,
    content_type: str | None,
    content: bytes,
) -> tuple[bool, dict[str, Any] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    if not file_name:
        return False, None, "file_name пустой"

    size_bytes = len(content or b"")
    sha256 = hashlib.sha256(content or b"").hexdigest()

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO task_files (
                        task_id, file_name, content_type, size_bytes, sha256, content
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, task_id, file_name, content_type, size_bytes, sha256, created_at;
                    """,
                    (task_id, file_name, content_type, size_bytes, sha256, content),
                )
                row = cur.fetchone()
            conn.commit()
        return True, _row_to_file(row), "OK"
    except psycopg.errors.UndefinedTable:
        return False, None, "schema_not_initialized"
    except psycopg.errors.ForeignKeyViolation:
        return False, None, "Задача не найдена"
    except Exception as e:
        return False, None, f"Ошибка сохранения файла: {e.__class__.__name__}"


def list_task_files(
    database_url: str | None,
    task_id: int,
) -> tuple[bool, list[dict[str, Any]] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, task_id, file_name, content_type, size_bytes, sha256, created_at
                    FROM task_files
                    WHERE task_id = %s
                    ORDER BY id ASC;
                    """,
                    (task_id,),
                )
                rows = cur.fetchall() or []

        return True, [_row_to_file(r) for r in rows], "OK"
    except psycopg.errors.UndefinedTable:
        return False, None, "schema_not_initialized"
    except Exception as e:
        return False, None, f"Ошибка списка файлов: {e.__class__.__name__}"


def get_task_file_content(
    database_url: str | None,
    task_id: int,
    file_id: int,
) -> tuple[bool, dict[str, Any] | None, bytes | None, str]:
    if not database_url:
        return False, None, None, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, task_id, file_name, content_type, size_bytes, sha256, created_at, content
                    FROM task_files
                    WHERE task_id = %s AND id = %s
                    LIMIT 1;
                    """,
                    (task_id, file_id),
                )
                row = cur.fetchone()

        if not row:
            return False, None, None, "Файл не найден"

        meta = {
            "id": row[0],
            "task_id": row[1],
            "file_name": row[2],
            "content_type": row[3],
            "size_bytes": int(row[4] or 0),
            "sha256": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
        }
        return True, meta, row[7] if row[7] is not None else b"", "OK"
    except psycopg.errors.UndefinedTable:
        return False, None, None, "schema_not_initialized"
    except Exception as e:
        return False, None, None, f"Ошибка чтения файла: {e.__class__.__name__}"
