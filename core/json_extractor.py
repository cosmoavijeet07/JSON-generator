import json
import re
from jsonschema import validate, ValidationError

def extract_json(output: str):
    # Try multiple extraction patterns
    patterns = [
        r"\{[\s\S]*\}",  # Original pattern
        r"```json\s*(\{[\s\S]*?\})\s*```",  # Code block pattern
        r"```\s*(\{[\s\S]*?\})\s*```",  # Generic code block
        r"(?:json|JSON)?\s*(\{[\s\S]*\})",  # With optional json prefix
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find any JSON-like structure
    lines = output.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        if '{' in line and not in_json:
            in_json = True
            brace_count = line.count('{') - line.count('}')
            json_lines.append(line)
        elif in_json:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                break
    
    if json_lines:
        try:
            return json.loads('\n'.join(json_lines))
        except json.JSONDecodeError:
            pass
    
    raise ValueError("No valid JSON found.")

def validate_against_schema(schema: dict, instance: dict):
    try:
        validate(instance=instance, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e)

def partial_validate_against_schema(schema: dict, instance: dict):
    """
    NEW: Validate allowing for partial/incomplete extractions
    """
    if not isinstance(instance, dict):
        return False, "Instance must be a dictionary"
    
    try:
        # Create a relaxed schema for partial validation
        relaxed_schema = create_relaxed_schema(schema)
        validate(instance=instance, schema=relaxed_schema)
        
        # Check how complete the extraction is
        completeness_score = calculate_completeness(schema, instance)
        
        return True, None, completeness_score
    except ValidationError as e:
        return False, str(e), 0.0

def create_relaxed_schema(schema: dict) -> dict:
    """Create a more lenient version of the schema for partial validation"""
    relaxed = schema.copy()
    
    # Remove required fields for partial validation
    if 'required' in relaxed:
        relaxed['required'] = []
    
    # Recursively relax nested objects
    if 'properties' in relaxed:
        for prop_name, prop_schema in relaxed['properties'].items():
            if isinstance(prop_schema, dict) and prop_schema.get('type') == 'object':
                relaxed['properties'][prop_name] = create_relaxed_schema(prop_schema)
    
    return relaxed

def calculate_completeness(schema: dict, instance: dict) -> float:
    """Calculate how complete the extraction is compared to the schema"""
    if not schema.get('properties'):
        return 1.0
    
    total_fields = len(schema['properties'])
    filled_fields = 0
    
    for field_name, field_schema in schema['properties'].items():
        if field_name in instance and instance[field_name] is not None:
            if isinstance(instance[field_name], (list, dict)):
                if instance[field_name]:  # Non-empty list or dict
                    filled_fields += 1
            elif isinstance(instance[field_name], str):
                if instance[field_name].strip():  # Non-empty string
                    filled_fields += 1
            else:
                filled_fields += 1
    
    return filled_fields / total_fields if total_fields > 0 else 1.0
