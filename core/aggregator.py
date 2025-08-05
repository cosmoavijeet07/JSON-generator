import copy

def merge_json_outputs(outputs, schema):
    result = {}
    for item in outputs:
        for key, value in item.items():
            if key not in result:
                result[key] = copy.deepcopy(value)
            else:
                if isinstance(result[key], list) and isinstance(value, list):
                    merged = result[key] + value
                    seen = set(); deduped = []
                    for v in merged:
                        h = str(v)
                        if h not in seen: deduped.append(v); seen.add(h)
                    result[key] = deduped
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_json_outputs([result[key], value], schema.get('properties', {}).get(key, {}))
                elif value is not None and result[key] in [None, '', []]:
                    result[key] = copy.deepcopy(value)
    return result
