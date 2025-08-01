from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import json

from core import (
    schema_validator,
    prompt_engine,
    llm_interface,
    json_extractor,
    logger_service,
    session_manager,
    token_estimator
)

app = FastAPI(title="AI JSON Extractor API")

@app.post("/extract-json/")
async def extract_json_api(
    schema: UploadFile = File(...),
    text: UploadFile = File(...),
    model: str = "gpt-4.1-2025-04-14"
):
    schema_str = (await schema.read()).decode()
    text_str = (await text.read()).decode()

    try:
        schema_json = json.loads(schema_str)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON schema: {e}"})

    # Validate schema
    valid, err = schema_validator.is_valid_schema(schema_json)
    if not valid:
        return JSONResponse(status_code=400, content={"error": f"Schema validation failed: {err}"})

    session_id = session_manager.create_session()
    logger_service.log(session_id, "model_used", model)
    logger_service.log(session_id, "schema", schema_str)
    logger_service.log(session_id, "text", text_str)

    token_count = token_estimator.estimate_tokens(schema_str + text_str)

    for attempt in range(3):
        prompt = prompt_engine.create_prompt(schema_str, text_str, error=err if attempt > 0 else None)
        logger_service.log(session_id, f"prompt_{attempt+1}", prompt)

        output = llm_interface.call_llm(prompt, model=model)
        logger_service.log(session_id, f"output_{attempt+1}", output)

        try:
            result = json_extractor.extract_json(output)
            valid, err = json_extractor.validate_against_schema(schema_json, result)

            if valid:
                final_path = f"logs/{session_id}/final_output.json"
                with open(final_path, "w") as f:
                    json.dump(result, f, indent=2)
                return {
                    "session_id": session_id,
                    "token_estimate": token_count,
                    "attempts": attempt + 1,
                    "json_result": result
                }

        except Exception as e:
            err = str(e)

    return JSONResponse(
        status_code=422,
        content={
            "error": f"Validation failed after 3 attempts: {err}",
            "session_id": session_id,
            "token_estimate": token_count
        }
    )


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
