import copy
import json
from core import logger_service

def merge_json_outputs(outputs, schema, session_id=None):
    """
    Enhanced merge function with detailed logging for post-processing validation
    """
    if session_id:
        logger_service.log(session_id, "pre_merge_outputs", json.dumps(outputs, indent=2))
    
    result = {}
    merge_conflicts = []
    
    for idx, item in enumerate(outputs):
        if not isinstance(item, dict):
            if session_id:
                logger_service.log(session_id, "merge_warning", f"Chunk {idx} output is not a dict: {type(item)}")
            continue
            
        for key, value in item.items():
            if key not in result:
                result[key] = copy.deepcopy(value)
            else:
                # Track merge conflicts for validation
                if result[key] != value:
                    merge_conflicts.append({
                        "field": key,
                        "existing": result[key],
                        "new": value,
                        "chunk": idx
                    })
                
                if isinstance(result[key], list) and isinstance(value, list):
                    merged = result[key] + value
                    seen = set()
                    deduped = []
                    for v in merged:
                        h = str(v)
                        if h not in seen:
                            deduped.append(v)
                            seen.add(h)
                    result[key] = deduped
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_json_outputs([result[key], value], 
                                                   schema.get('properties', {}).get(key, {}), 
                                                   session_id)
                elif value is not None and result[key] in [None, '', []]:
                    result[key] = copy.deepcopy(value)
    
    if session_id and merge_conflicts:
        logger_service.log(session_id, "merge_conflicts", json.dumps(merge_conflicts, indent=2))
    
    return result
