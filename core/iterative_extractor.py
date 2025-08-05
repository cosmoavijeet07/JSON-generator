import json

def iterative_schema_extract(llm_fn, prompt_engine, schema, chunks, validator_fn, max_passes=2):
    all_results = []
    for chunk in chunks:
        partial = None; errors = None
        for attempt in range(max_passes):
            prompt = prompt_engine.create_advanced_prompt(schema, chunk, error=errors, pass_num=attempt+1)
            out = llm_fn(prompt)
            try:
                partial = json.loads(out) if isinstance(out, str) else out
                valid, errors = validator_fn(schema, partial)
                if valid: break
            except Exception as e:
                errors = str(e)
        all_results.append(partial or {})
    return all_results
