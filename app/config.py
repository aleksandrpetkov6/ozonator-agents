import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    redis_url: str | None
    az_run_task_base_url: str
    as_run_task_base_url: str
    ak_run_task_base_url: str
    aa_auto_orchestration_enabled: bool
    aa_max_rework_cycles: int



def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL"),
        redis_url=os.getenv("REDIS_URL"),
        az_run_task_base_url=(
            os.getenv("AZ_RUN_TASK_BASE_URL") or "https://ozonator-az-dev.onrender.com"
        ).rstrip("/"),
        as_run_task_base_url=(
            os.getenv("AS_RUN_TASK_BASE_URL") or "https://ozonator-as-dev.onrender.com"
        ).rstrip("/"),
        ak_run_task_base_url=(
            os.getenv("AK_RUN_TASK_BASE_URL") or "https://ozonator-ak-dev.onrender.com"
        ).rstrip("/"),
        aa_auto_orchestration_enabled=_env_bool("AA_AUTO_ORCHESTRATION_ENABLED", True),
        aa_max_rework_cycles=max(1, int(os.getenv("AA_MAX_REWORK_CYCLES") or "2")),
    )
