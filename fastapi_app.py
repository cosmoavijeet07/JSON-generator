from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, json, asyncio

from core import (
    schema_validator,
    prompt_engine,
    llm_interface,
    json_extractor,
    logger_service,
    session_manager,
    token_estimator,
    semantic_chunker,
    aggregator,
    iterative_extractor,
    preprocessor,
    post_processor
)

app = FastAPI(title="AI JSON Extractor API")

@app.post("/extract-json/")
async def extract_json_api(
    schema: UploadFile = File(...),
    text: UploadFile = File(...),
    model: str = "gpt-4.1-2025-04-14",
    rectification_attempts: int = 3,
    chunk_size: int = 3500,
    overlap: int = 200,
    max_passes: int = 2
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
    
    # Create session
    session_id = session_manager.create_session()
    logger_service.log(session_id, "model_used", model)
    
    # Preprocess and chunk
    text_cleaned = preprocessor.clean_ocr(text_str)
    chunks = semantic_chunker.split_semantic(text_cleaned, max_len=chunk_size, overlap=overlap)
    
    def llm_fn(prompt):
        return llm_interface.call_llm(prompt, model=model)
    
    # Iterative extraction
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, iterative_extractor.iterative_schema_extract,
        llm_fn, prompt_engine, schema_str, chunks, json_extractor.validate_against_schema, max_passes
    )
    
    # Aggregate results
    preliminary_json = aggregator.merge_json_outputs(results, schema_json, session_id)
    
    # Post-processing validation and rectification
    final_json, validation_success, validation_error = post_processor.validate_and_rectify_final_json(
        final_json=preliminary_json,
        schema=schema_str,
        original_chunks=chunks,
        llm_fn=llm_fn,
        session_id=session_id,
        max_rectification_attempts=rectification_attempts
    )
    
    # Save final output
    final_path = f"logs/{session_id}/final_output.json"
    with open(final_path, "w") as f:
        json.dump(final_json, f, indent=2)
    
    return {
        "session_id": session_id,
        "chunks": len(chunks),
        "json_result": final_json,
        "validation_successful": validation_success,
        "validation_error": validation_error,
        "rectification_attempts_used": rectification_attempts,
        "token_estimate": token_estimator.estimate_tokens(schema_str + text_str, model)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI JSON Extractor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
