@app.post("/aa/run-task/{task_id}")
def aa_run_task(task_id: int):
    settings = get_settings()

    ok, task, message = get_task_record(settings.database_url, task_id)
    if not ok:
        return JSONResponse(
            status_code=404 if message == "Задача не найдена" else 503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_run_started",
        level="info",
        message="AA начал обработку задачи",
        meta={
            "target_agent": task["target_agent"],
            "task_type": task["task_type"],
        },
    )

    ok, task, message = update_task_status(settings.database_url, task_id, "in_progress")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    payload = task.get("payload") or {}

    if task.get("task_type") == "ozonator_inventory_locations_fix":
        execution_result = _build_az_brief_for_inventory_locations_fix(task_id, payload)
    else:
        execution_result = {
            "routed_to": "AZ",
            "mode": "aa_handoff_v1",
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "handoff_ready": True,
            "next_agent": "AZ",
            "aa_status": "routed_to_az",
            "note": "AA подготовил задачу и передал handoff в AZ",
        }

    if isinstance(execution_result, dict):
        execution_result["handoff_ready"] = True
        execution_result["next_agent"] = "AZ"
        execution_result["aa_executor"] = "AA"

    ok_set, task, message_set = set_task_result(
        settings.database_url,
        task_id=task_id,
        result=execution_result,
        error_message=None,
    )
    if not ok_set:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message_set,
                "task": None,
            },
        )

    ok, task, message = update_task_status(settings.database_url, task_id, "AA_ROUTED")
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "service": "AA",
                "operation": "aa_run_task",
                "status": "error",
                "message": message,
                "task": None,
            },
        )

    write_orchestration_log(
        settings.database_url,
        task_id=task_id,
        actor_agent="AA",
        event_type="task_routed_to_az",
        level="info",
        message="AA подготовил handoff и маршрутизировал задачу в AZ",
        meta=execution_result,
    )

    return JSONResponse(
        status_code=200,
        content={
            "service": "AA",
            "operation": "aa_run_task",
            "status": "ok",
            "message": "AA подготовил handoff в AZ",
            "task": task,
            "execution_result": execution_result,
        },
    )
