def schema_branches(schema_obj, prefix=""):
    results = []
    if schema_obj.get("type") == "object":
        for k, v in schema_obj.get("properties", {}).items():
            fullpath = f"{prefix}.{k}" if prefix else k
            results.append((fullpath, v))
            results += schema_branches(v, prefix=fullpath)
    return results
