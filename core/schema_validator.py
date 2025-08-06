import json
from jsonschema import Draft7Validator, validate, ValidationError
from typing import Dict, Any, Tuple, List
import jsonref

class SchemaValidator:
    def __init__(self):
        self.validator = None
        self.schema_cache = {}
    
    def validate_schema(self, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate if the schema itself is valid"""
        try:
            # Resolve any $ref references
            resolved_schema = jsonref.JsonRef.replace_refs(schema)
            Draft7Validator.check_schema(resolved_schema)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def validate_instance(self, instance: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate JSON instance against schema"""
        try:
            validate(instance=instance, schema=schema)
            return True, None
        except ValidationError as e:
            return False, self._format_validation_error(e)
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """Format validation error for better readability"""
        path = " -> ".join(str(p) for p in error.path) if error.path else "root"
        return f"Validation error at {path}: {error.message}"
    
    def analyze_schema_complexity(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema complexity metrics"""
        metrics = {
            "max_depth": self._get_max_depth(schema),
            "total_fields": self._count_fields(schema),
            "required_fields": len(schema.get("required", [])),
            "has_nested_objects": self._has_nested_objects(schema),
            "has_arrays": self._has_arrays(schema),
            "has_references": "$ref" in str(schema),
            "estimated_complexity": 0
        }
        
        # Calculate complexity score
        metrics["estimated_complexity"] = (
            metrics["max_depth"] * 2 +
            metrics["total_fields"] * 0.5 +
            (5 if metrics["has_nested_objects"] else 0) +
            (3 if metrics["has_arrays"] else 0)
        )
        
        return metrics
    
    def _get_max_depth(self, schema: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if not isinstance(schema, dict):
            return current_depth
        
        max_depth = current_depth
        
        if "properties" in schema:
            for prop in schema["properties"].values():
                depth = self._get_max_depth(prop, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        if "items" in schema:
            depth = self._get_max_depth(schema["items"], current_depth + 1)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_fields(self, schema: Dict[str, Any]) -> int:
        """Count total number of fields in schema"""
        if not isinstance(schema, dict):
            return 0
        
        count = 0
        if "properties" in schema:
            count += len(schema["properties"])
            for prop in schema["properties"].values():
                count += self._count_fields(prop)
        
        if "items" in schema:
            count += self._count_fields(schema["items"])
        
        return count
    
    def _has_nested_objects(self, schema: Dict[str, Any]) -> bool:
        """Check if schema has nested objects"""
        if not isinstance(schema, dict):
            return False
        
        if schema.get("type") == "object" and "properties" in schema:
            for prop in schema["properties"].values():
                if isinstance(prop, dict) and prop.get("type") == "object":
                    return True
                if self._has_nested_objects(prop):
                    return True
        
        return False
    
    def _has_arrays(self, schema: Dict[str, Any]) -> bool:
        """Check if schema has arrays"""
        if not isinstance(schema, dict):
            return False
        
        if schema.get("type") == "array":
            return True
        
        if "properties" in schema:
            for prop in schema["properties"].values():
                if self._has_arrays(prop):
                    return True
        
        return False