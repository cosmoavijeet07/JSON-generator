import json
import re
from typing import Dict, Any, Optional, List, Tuple

class JSONExtractor:
    def __init__(self):
        self.extraction_patterns = [
            r'\{[\s\S]*\}',  # Standard JSON object
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',  # Generic code block
        ]
    
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM output"""
        # Try each pattern
        for pattern in self.extraction_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean the match
                    cleaned = self._clean_json_string(match)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        # If no patterns work, try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("No valid JSON found in the output")
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string for parsing"""
        # Remove common artifacts
        json_str = json_str.strip()
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        json_str = re.sub(r'^```\s*', '', json_str)
        
        # Fix common issues
        json_str = json_str.replace("'", '"')  # Replace single quotes
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def merge_json_outputs(self, outputs: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge multiple JSON outputs"""
        if not outputs:
            return {}
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Start with the first output as base
        merged = outputs[0].copy()
        
        for output in outputs[1:]:
            merged = self._deep_merge(merged, output, schema)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two JSON objects based on schema"""
        result = base.copy()
        
        for key, value in update.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Recursively merge nested objects
                result[key] = self._deep_merge(result[key], value, schema.get("properties", {}).get(key, {}))
            elif isinstance(value, list) and isinstance(result[key], list):
                # Merge arrays intelligently
                result[key] = self._merge_arrays(result[key], value)
            elif value is not None and result[key] is None:
                # Override null values
                result[key] = value
        
        return result
    
    def _merge_arrays(self, arr1: List, arr2: List) -> List:
        """Merge two arrays, avoiding duplicates"""
        # Simple deduplication based on string representation
        seen = set()
        result = []
        
        for item in arr1 + arr2:
            item_str = json.dumps(item, sort_keys=True)
            if item_str not in seen:
                seen.add(item_str)
                result.append(item)
        
        return result
