import json
from typing import Any

import psycopg


def _row_to_task(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "external_task_id": row[1],
        "source_agent": row[2],
        "target_agent": row[3],
        "task_type": row[4],
        "status": row[5],
        "priority": row[6],
        "payload": row[7] if row[7] is not None else {},
        "result": row[8] if row[8] is not None else None,
        "error_message": row[9],
        "parent_task_id": row[10],
        "created_at": row[11].isoformat() if row[11] else None,
        "updated_at": row[12].isoformat() if row[12] else None,
    }


def create_task_record(
    database_url: str | None,
    task_data: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    payload_json = json.dumps(task_data.get("payload") or {}, ensure_ascii=False)

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (
                        external_task_id,
                        source_agent,
                        target_agent,
                        task_type,
                        status,
                        priority,
                        payload,
                        parent_task_id
                    )
                    VALUES (%s, %s, %s, %s, 'new', %s, %s::jsonb, %s)
                    RETURNING
                        id, external_task_id, source_agent, target_agent, task_type, status,
                        priority, payload, result, error_message, parent_task_id, created_at, updated_at;
                    """,
                    (
                        task_data.get("external_task_id"),
                        task_data.get("source_agent") or "AA",
                        task_data.get("target_agent"),
                        task_data.get("task_type"),
                        int(task_data.get("priority") or 100),
                        payload_json,
                        task_data.get("parent_task_id"),
                    ),
                )
                row = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO orchestration_logs (
                        task_id, actor_agent, event_type, level, message, meta
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    (
                        row[0],
                        task_data.get("source_agent") or "AA",
                        "task_created",
                        "info",
                        "Задача создана через POST /tasks/create",
                        json.dumps(
                            {
                                "target_agent": row[3],
                                "task_type": row[4],
                                "status": row[5],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

            conn.commit()

        return True, _row_to_task(row), "Задача создана"

    except psycopg.errors.UniqueViolation:
        return False, None, "external_task_id уже существует"
    except Exception as e:
        return False, None, f"Ошибка создания задачи: {e.__class__.__name__}"


def get_task_record(
    database_url: str | None,
    task_id: int,
) -> tuple[bool, dict[str, Any] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, external_task_id, source_agent, target_agent, task_type, status,
                        priority, payload, result, error_message, parent_task_id, created_at, updated_at
                    FROM tasks
                    WHERE id = %s;
                    """,
                    (task_id,),
                )
                row = cur.fetchone()

        if not row:
            return False, None, "Задача не найдена"

        return True, _row_to_task(row), "OK"

    except Exception as e:
        return False, None, f"Ошибка чтения задачи: {e.__class__.__name__}"


def update_task_status(
    database_url: str | None,
    task_id: int,
    new_status: str,
) -> tuple[bool, dict[str, Any] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                    RETURNING
                        id, external_task_id, source_agent, target_agent, task_type, status,
                        priority, payload, result, error_message, parent_task_id, created_at, updated_at;
                    """,
                    (new_status, task_id),
                )
                row = cur.fetchone()

            conn.commit()

        if not row:
            return False, None, "Задача не найдена"

        return True, _row_to_task(row), "Статус обновлён"

    except Exception as e:
        return False, None, f"Ошибка обновления статуса: {e.__class__.__name__}"


def set_task_result(
    database_url: str | None,
    task_id: int,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> tuple[bool, dict[str, Any] | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    try:
        result_json = json.dumps(result or {}, ensure_ascii=False) if result is not None else None

        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET result = %s::jsonb,
                        error_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING
                        id, external_task_id, source_agent, target_agent, task_type, status,
                        priority, payload, result, error_message, parent_task_id, created_at, updated_at;
                    """,
                    (result_json, error_message, task_id),
                )
                row = cur.fetchone()

            conn.commit()

        if not row:
            return False, None, "Задача не найдена"

        return True, _row_to_task(row), "Результат задачи сохранён"

    except Exception as e:
        return False, None, f"Ошибка сохранения результата: {e.__class__.__name__}"


def write_orchestration_log(
    database_url: str | None,
    task_id: int,
    actor_agent: str,
    event_type: str,
    level: str,
    message: str,
    meta: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if not database_url:
        return False, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO orchestration_logs (
                        task_id, actor_agent, event_type, level, message, meta
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    (
                        task_id,
                        actor_agent,
                        event_type,
                        level,
                        message,
                        json.dumps(meta or {}, ensure_ascii=False),
                    ),
                )

            conn.commit()

        return True, "OK"

    except Exception as e:
        return False, f"Ошибка записи лога: {e.__class__.__name__}"


def get_task_logs(
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
                    SELECT
                        id, task_id, actor_agent, event_type, level, message, meta, created_at
                    FROM orchestration_logs
                    WHERE task_id = %s
                    ORDER BY id ASC;
                    """,
                    (task_id,),
                )
                rows = cur.fetchall()

        logs: list[dict[str, Any]] = []
        for row in rows:
            logs.append(
                {
                    "id": row[0],
                    "task_id": row[1],
                    "actor_agent": row[2],
                    "event_type": row[3],
                    "level": row[4],
                    "message": row[5],
                    "meta": row[6] if row[6] is not None else {},
                    "created_at": row[7].isoformat() if row[7] else None,
                }
            )

        return True, logs, "OK"

    except Exception as e:
        return False, None, f"Ошибка чтения логов: {e.__class__.__name__}"
