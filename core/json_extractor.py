import json
import re
from jsonschema import validate, ValidationError

def extract_json(output: str):
    match = re.search(r"\{[\s\S]*\}", output)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found.")

def validate_against_schema(schema: dict, instance: dict):
    try:
        validate(instance=instance, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e)
