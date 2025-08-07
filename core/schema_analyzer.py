from jsonschema import Draft7Validator, exceptions as jsonschema_exceptions
from typing import Dict, Any, Tuple, List, Optional
import json

def is_valid_schema(schema_json: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate if a JSON schema is valid according to Draft 7
    
    Args:
        schema_json: JSON schema to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        Draft7Validator.check_schema(schema_json)
        return True, None
    except jsonschema_exceptions.SchemaError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error validating schema: {str(e)}"


def analyze_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a JSON schema and return statistics
    
    Args:
        schema: JSON schema to analyze
    
    Returns:
        Dictionary with schema statistics
    """
    stats = {
        'type': schema.get('type', 'unknown'),
        'has_properties': 'properties' in schema,
        'property_count': 0,
        'required_count': 0,
        'max_depth': 0,
        'has_definitions': '$defs' in schema or 'definitions' in schema,
        'has_patterns': 'patternProperties' in schema,
        'has_conditionals': any(k in schema for k in ['if', 'then', 'else', 'allOf', 'anyOf', 'oneOf']),
        'has_array_items': 'items' in schema,
        'complexity': 'simple'
    }
    
    # Count properties
    if 'properties' in schema:
        stats['property_count'] = len(schema['properties'])
    
    # Count required fields
    if 'required' in schema:
        stats['required_count'] = len(schema['required'])
    
    # Calculate max depth
    stats['max_depth'] = calculate_schema_depth(schema)
    
    # Determine complexity
    if stats['max_depth'] > 5 or stats['property_count'] > 50:
        stats['complexity'] = 'very_complex'
    elif stats['max_depth'] > 3 or stats['property_count'] > 20:
        stats['complexity'] = 'complex'
    elif stats['max_depth'] > 2 or stats['property_count'] > 10:
        stats['complexity'] = 'moderate'
    
    return stats


def calculate_schema_depth(schema: Dict[str, Any], current_depth: int = 0) -> int:
    """
    Calculate the maximum nesting depth of a schema
    
    Args:
        schema: JSON schema
        current_depth: Current recursion depth
    
    Returns:
        Maximum depth found
    """
    if current_depth > 20:  # Prevent infinite recursion
        return current_depth
    
    max_depth = current_depth
    
    # Check properties
    if 'properties' in schema and isinstance(schema['properties'], dict):
        for prop_schema in schema['properties'].values():
            if isinstance(prop_schema, dict):
                depth = calculate_schema_depth(prop_schema, current_depth + 1)
                max_depth = max(max_depth, depth)
    
    # Check array items
    if 'items' in schema:
        if isinstance(schema['items'], dict):
            depth = calculate_schema_depth(schema['items'], current_depth + 1)
            max_depth = max(max_depth, depth)
        elif isinstance(schema['items'], list):
            for item_schema in schema['items']:
                if isinstance(item_schema, dict):
                    depth = calculate_schema_depth(item_schema, current_depth + 1)
                    max_depth = max(max_depth, depth)
    
    # Check additional properties
    if 'additionalProperties' in schema and isinstance(schema['additionalProperties'], dict):
        depth = calculate_schema_depth(schema['additionalProperties'], current_depth + 1)
        max_depth = max(max_depth, depth)
    
    # Check pattern properties
    if 'patternProperties' in schema and isinstance(schema['patternProperties'], dict):
        for pattern_schema in schema['patternProperties'].values():
            if isinstance(pattern_schema, dict):
                depth = calculate_schema_depth(pattern_schema, current_depth + 1)
                max_depth = max(max_depth, depth)
    
    # Check conditionals
    for key in ['if', 'then', 'else']:
        if key in schema and isinstance(schema[key], dict):
            depth = calculate_schema_depth(schema[key], current_depth + 1)
            max_depth = max(max_depth, depth)
    
    # Check combiners
    for key in ['allOf', 'anyOf', 'oneOf']:
        if key in schema and isinstance(schema[key], list):
            for sub_schema in schema[key]:
                if isinstance(sub_schema, dict):
                    depth = calculate_schema_depth(sub_schema, current_depth + 1)
                    max_depth = max(max_depth, depth)
    
    return max_depth


def simplify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify a schema by removing non-essential properties
    
    Args:
        schema: Original JSON schema
    
    Returns:
        Simplified schema
    """
    essential_keys = {
        'type', 'properties', 'required', 'items', 'enum',
        'minimum', 'maximum', 'minLength', 'maxLength',
        'pattern', 'format', 'minItems', 'maxItems',
        'uniqueItems', 'additionalProperties'
    }
    
    simplified = {}
    
    for key, value in schema.items():
        if key in essential_keys:
            if key == 'properties' and isinstance(value, dict):
                # Recursively simplify properties
                simplified[key] = {
                    prop_name: simplify_schema(prop_schema) if isinstance(prop_schema, dict) else prop_schema
                    for prop_name, prop_schema in value.items()
                }
            elif key == 'items':
                if isinstance(value, dict):
                    simplified[key] = simplify_schema(value)
                else:
                    simplified[key] = value
            else:
                simplified[key] = value
    
    return simplified


def extract_required_fields(schema: Dict[str, Any], path: str = "") -> List[str]:
    """
    Extract all required field paths from a schema
    
    Args:
        schema: JSON schema
        path: Current path in the schema
    
    Returns:
        List of required field paths
    """
    required_fields = []
    
    # Get required fields at current level
    if 'required' in schema and isinstance(schema['required'], list):
        for field in schema['required']:
            field_path = f"{path}.{field}" if path else field
            required_fields.append(field_path)
    
    # Recursively check properties
    if 'properties' in schema and isinstance(schema['properties'], dict):
        for prop_name, prop_schema in schema['properties'].items():
            prop_path = f"{path}.{prop_name}" if path else prop_name
            if isinstance(prop_schema, dict):
                required_fields.extend(extract_required_fields(prop_schema, prop_path))
    
    # Check array items
    if 'items' in schema and isinstance(schema['items'], dict):
        items_path = f"{path}[]" if path else "[]"
        required_fields.extend(extract_required_fields(schema['items'], items_path))
    
    return required_fields


def get_field_types(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract field types from a schema
    
    Args:
        schema: JSON schema
    
    Returns:
        Dictionary mapping field names to types
    """
    field_types = {}
    
    if 'properties' in schema and isinstance(schema['properties'], dict):
        for prop_name, prop_schema in schema['properties'].items():
            if isinstance(prop_schema, dict):
                field_types[prop_name] = prop_schema.get('type', 'any')
                
                # Recursively get nested field types
                if prop_schema.get('type') == 'object':
                    nested_types = get_field_types(prop_schema)
                    for nested_name, nested_type in nested_types.items():
                        field_types[f"{prop_name}.{nested_name}"] = nested_type
    
    return field_types


def validate_json_string(json_string: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a string is valid JSON
    
    Args:
        json_string: String to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        json.loads(json_string)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
    except Exception as e:
        return False, f"Error parsing JSON: {str(e)}"


def create_minimal_schema(sample_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a minimal schema from a sample JSON
    
    Args:
        sample_json: Sample JSON object
    
    Returns:
        Generated JSON schema
    """
    def infer_type(value: Any) -> Dict[str, Any]:
        """Infer schema type from a value"""
        if value is None:
            return {"type": "null"}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, list):
            if not value:
                return {"type": "array", "items": {}}
            else:
                # Infer items type from first element
                return {"type": "array", "items": infer_type(value[0])}
        elif isinstance(value, dict):
            properties = {}
            required = []
            
            for key, val in value.items():
                properties[key] = infer_type(val)
                if val is not None:
                    required.append(key)
            
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            
            return schema
        else:
            return {}
    
    return infer_type(sample_json)