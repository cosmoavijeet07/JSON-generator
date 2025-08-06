import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

class MergeManager:
    def __init__(self):
        self.merge_strategies = {
            "simple": self._simple_merge,
            "voting": self._voting_merge,
            "confidence": self._confidence_merge,
            "hierarchical": self._hierarchical_merge
        }
    
    def merge_outputs(self, 
                     outputs: List[Dict[str, Any]], 
                     schema: Dict[str, Any],
                     strategy: str = "voting") -> Dict[str, Any]:
        """Merge multiple JSON outputs using specified strategy"""
        
        if not outputs:
            return {}
        
        if len(outputs) == 1:
            return outputs[0]
        
        merge_func = self.merge_strategies.get(strategy, self._voting_merge)
        return merge_func(outputs, schema)
    
    def _simple_merge(self, outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simple merge - take first non-null value"""
        merged = {}
        
        for output in outputs:
            for key, value in output.items():
                if key not in merged or merged[key] is None:
                    merged[key] = value
        
        return merged
    
    def _voting_merge(self, outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge by voting - most common value wins"""
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for output in outputs:
            all_keys.update(output.keys())
        
        for key in all_keys:
            values = []
            for output in outputs:
                if key in output and output[key] is not None:
                    values.append(json.dumps(output[key], sort_keys=True))
            
            if values:
                # Find most common value
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[v] += 1
                
                most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                merged[key] = json.loads(most_common)
        
        return merged
    
    def _confidence_merge(self, outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge based on confidence scores (if available)"""
        # For now, fall back to voting merge
        # In a real implementation, outputs would include confidence scores
        return self._voting_merge(outputs, schema)
    
    def _hierarchical_merge(self, outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Hierarchical merge - respects schema structure"""
        merged = {}
        
        # Process required fields first
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            values = [output.get(field) for output in outputs if field in output]
            values = [v for v in values if v is not None]
            
            if values:
                # For required fields, take the most complete value
                merged[field] = self._select_most_complete(values)
        
        # Process optional fields
        all_keys = set()
        for output in outputs:
            all_keys.update(output.keys())
        
        optional_fields = all_keys - set(required_fields)
        
        for field in optional_fields:
            values = [output.get(field) for output in outputs if field in output]
            values = [v for v in values if v is not None]
            
            if values:
                merged[field] = self._select_most_complete(values)
        
        return merged
    
    def _select_most_complete(self, values: List[Any]) -> Any:
        """Select the most complete value from a list"""
        if not values:
            return None
        
        # Sort by completeness
        def completeness_score(value):
            if value is None:
                return 0
            elif isinstance(value, dict):
                return sum(1 for v in value.values() if v is not None)
            elif isinstance(value, list):
                return len(value)
            elif isinstance(value, str):
                return len(value)
            else:
                return 1
        
        return max(values, key=completeness_score)
    
    def validate_merge(self, merged: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate merged output against schema"""
        errors = []
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in merged or merged[field] is None:
                errors.append(f"Required field '{field}' is missing")
        
        # Check types
        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in merged and merged[field] is not None:
                    if not self._check_type(merged[field], field_schema):
                        errors.append(f"Field '{field}' has incorrect type")
        
        return len(errors) == 0, errors
    
    def _check_type(self, value: Any, schema: Dict[str, Any]) -> bool:
        """Check if value matches schema type"""
        expected_type = schema.get("type")
        
        if not expected_type:
            return True
        
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        
        return True