import psycopg


CORE_AGENT_INSTRUCTIONS = [
    {
        "agent_code": "AA",
        "version": 1,
        "title": "CORE:AA:ORCHESTRATOR",
        "instruction_text": (
            "АА — оркестратор. Принимает задачи, определяет target_agent, "
            "создает запись в tasks, логирует ключевые события в orchestration_logs."
        ),
    },
    {
        "agent_code": "AZ",
        "version": 1,
        "title": "CORE:AZ:UNIT_ECONOMICS",
        "instruction_text": (
            "АЗ — агент юнит-экономики. Считает показатели, формирует результат "
            "в структурированном виде и возвращает итог в result/payload задачи."
        ),
    },
    {
        "agent_code": "AS",
        "version": 1,
        "title": "CORE:AS:ANALYST",
        "instruction_text": (
            "АС — аналитик/проверяющий. Делает аналитическую проверку, формирует "
            "qa_reports и qa_findings по заданным правилам."
        ),
    },
    {
        "agent_code": "AK",
        "version": 1,
        "title": "CORE:AK:CONTROLLER",
        "instruction_text": (
            "АК — контролер качества. Проверяет полноту, корректность и соответствие "
            "формату, фиксирует замечания и статус финальной проверки."
        ),
    },
]

CORE_COMMUNICATION_RULES = [
    {
        "rule_key": "global.no_secrets_in_logs",
        "rule_text": "Не писать секреты/ключи/DSN в ответы и логи.",
        "scope": "global",
    },
    {
        "rule_key": "global.json_structured_payloads",
        "rule_text": "Межагентное взаимодействие вести через структурированные JSON payload/result.",
        "scope": "global",
    },
    {
        "rule_key": "global.log_key_events",
        "rule_text": "Ключевые действия фиксировать в orchestration_logs с event_type и level.",
        "scope": "global",
    },
    {
        "rule_key": "aa.route_tasks",
        "rule_text": "AA обязан указывать source_agent, target_agent, task_type и status при создании задач.",
        "scope": "AA",
    },
    {
        "rule_key": "qa.findings_severity_required",
        "rule_text": "Каждое замечание в qa_findings должно иметь severity и finding_text.",
        "scope": "AS",
    },
    {
        "rule_key": "ak.final_decision_traceable",
        "rule_text": "Финальное решение AK должно быть трассируемо через qa_reports и orchestration_logs.",
        "scope": "AK",
    },
]


def seed_core_data(database_url: str | None) -> tuple[bool, dict | None, str]:
    if not database_url:
        return False, None, "DATABASE_URL не задан"

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                inserted_instructions = 0

                # agent_instructions: вставляем только если такой (agent_code + version + title) еще не существует
                for item in CORE_AGENT_INSTRUCTIONS:
                    cur.execute(
                        """
                        INSERT INTO agent_instructions (agent_code, version, title, instruction_text, is_active)
                        SELECT %s, %s, %s, %s, TRUE
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM agent_instructions
                            WHERE agent_code = %s
                              AND version = %s
                              AND title = %s
                        );
                        """,
                        (
                            item["agent_code"],
                            item["version"],
                            item["title"],
                            item["instruction_text"],
                            item["agent_code"],
                            item["version"],
                            item["title"],
                        ),
                    )
                    inserted_instructions += cur.rowcount

                upserted_rules = 0

                # communication_rules: idempotent через ON CONFLICT(rule_key)
                for rule in CORE_COMMUNICATION_RULES:
                    cur.execute(
                        """
                        INSERT INTO communication_rules (rule_key, rule_text, scope, is_active)
                        VALUES (%s, %s, %s, TRUE)
                        ON CONFLICT (rule_key)
                        DO UPDATE SET
                            rule_text = EXCLUDED.rule_text,
                            scope = EXCLUDED.scope,
                            is_active = TRUE,
                            updated_at = NOW();
                        """,
                        (rule["rule_key"], rule["rule_text"], rule["scope"]),
                    )
                    upserted_rules += 1

            conn.commit()

        details = {
            "agent_instructions_inserted": inserted_instructions,
            "communication_rules_upserted": upserted_rules,
        }
        return True, details, "Core seed completed"

    except Exception as e:
        return False, None, f"Ошибка заполнения стартовых данных: {e.__class__.__name__}"
