-- Базовая схема для контура АА / АЗ / АС / АК
-- Шаг 2: только подготовка схемы (без мигратора)

-- 1) tasks: центральная таблица задач между агентами
CREATE TABLE IF NOT EXISTS tasks (
    id BIGSERIAL PRIMARY KEY,
    external_task_id TEXT,
    source_agent TEXT,              -- кто поставил задачу (AA/AK/AS/AZ)
    target_agent TEXT,              -- кому задача адресована
    task_type TEXT NOT NULL,        -- тип задачи (qa, finance_check, summary, etc.)
    status TEXT NOT NULL DEFAULT 'new', -- new / queued / in_progress / done / failed
    priority INTEGER NOT NULL DEFAULT 100,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error_message TEXT,
    parent_task_id BIGINT REFERENCES tasks(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_tasks_external_task_id
    ON tasks(external_task_id)
    WHERE external_task_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_tasks_status
    ON tasks(status);

CREATE INDEX IF NOT EXISTS ix_tasks_target_agent
    ON tasks(target_agent);

CREATE INDEX IF NOT EXISTS ix_tasks_created_at
    ON tasks(created_at);


-- 2) qa_reports: итоговые QA-отчёты по задаче
CREATE TABLE IF NOT EXISTS qa_reports (
    id BIGSERIAL PRIMARY KEY,
    task_id BIGINT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    agent_code TEXT NOT NULL,       -- например AS (аналитик) / AK (контролёр)
    report_status TEXT NOT NULL DEFAULT 'draft', -- draft / final
    score NUMERIC(6,2),             -- агрегированный балл/оценка
    summary TEXT,                   -- краткое резюме
    raw_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_qa_reports_task_id
    ON qa_reports(task_id);

CREATE INDEX IF NOT EXISTS ix_qa_reports_agent_code
    ON qa_reports(agent_code);

CREATE INDEX IF NOT EXISTS ix_qa_reports_created_at
    ON qa_reports(created_at);


-- 3) qa_findings: замечания внутри QA-отчёта
CREATE TABLE IF NOT EXISTS qa_findings (
    id BIGSERIAL PRIMARY KEY,
    qa_report_id BIGINT NOT NULL REFERENCES qa_reports(id) ON DELETE CASCADE,
    severity TEXT NOT NULL,         -- low / medium / high / critical
    finding_code TEXT,              -- код правила/ошибки
    finding_text TEXT NOT NULL,     -- текст замечания
    recommendation TEXT,            -- что исправить
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_qa_findings_qa_report_id
    ON qa_findings(qa_report_id);

CREATE INDEX IF NOT EXISTS ix_qa_findings_severity
    ON qa_findings(severity);


-- 4) agent_instructions: инструкции для агентов (ДИ)
CREATE TABLE IF NOT EXISTS agent_instructions (
    id BIGSERIAL PRIMARY KEY,
    agent_code TEXT NOT NULL,       -- AA / AZ / AS / AK
    version INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    instruction_text TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_agent_instructions_agent_code
    ON agent_instructions(agent_code);

CREATE INDEX IF NOT EXISTS ix_agent_instructions_is_active
    ON agent_instructions(is_active);


-- 5) communication_rules: правила коммуникации (ФК)
CREATE TABLE IF NOT EXISTS communication_rules (
    id BIGSERIAL PRIMARY KEY,
    rule_key TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    scope TEXT,                     -- global / AA / AZ / AS / AK
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_communication_rules_rule_key
    ON communication_rules(rule_key);

CREATE INDEX IF NOT EXISTS ix_communication_rules_scope
    ON communication_rules(scope);

CREATE INDEX IF NOT EXISTS ix_communication_rules_is_active
    ON communication_rules(is_active);


-- 6) orchestration_logs: журнал оркестрации
CREATE TABLE IF NOT EXISTS orchestration_logs (
    id BIGSERIAL PRIMARY KEY,
    task_id BIGINT REFERENCES tasks(id) ON DELETE SET NULL,
    actor_agent TEXT,               -- кто записал событие
    event_type TEXT NOT NULL,       -- task_created / task_sent / qa_done / error / etc.
    level TEXT NOT NULL DEFAULT 'info', -- debug / info / warning / error
    message TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_orchestration_logs_task_id
    ON orchestration_logs(task_id);

CREATE INDEX IF NOT EXISTS ix_orchestration_logs_event_type
    ON orchestration_logs(event_type);

CREATE INDEX IF NOT EXISTS ix_orchestration_logs_created_at
    ON orchestration_logs(created_at);
