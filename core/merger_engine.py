import json
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
import copy
from .llm_interface import call_llm
from .json_extractor import extract_json
from .logger_service import log

class MergerEngine:
    """Handles intelligent merging of multiple JSON extractions using programmatic approaches"""
    
    def __init__(self):
        self.merge_strategies = {
            'scalar': self._merge_scalar,
            'array': self._merge_array,
            'object': self._merge_object
        }
    
    def intelligent_merge(
        self,
        extractions: List[Dict],
        target_schema: Dict,
        context: Dict,
        model: str
    ) -> Dict:
        """
        Intelligently merge multiple extractions programmatically
        
        Args:
            extractions: List of extraction results with chunk/partition info
            target_schema: The complete target schema
            context: Global context including embeddings
            model: The LLM model to use (only for fallback)
        
        Returns:
            Merged JSON matching the target schema
        """
        
        if not extractions:
            return self._create_empty_from_schema(target_schema)
        
        # Group extractions by partition
        partitioned_data = self._group_by_partition(extractions)
        
        # Merge within each partition across chunks
        partition_merges = {}
        for partition_idx, partition_extractions in partitioned_data.items():
            if len(partition_extractions) == 1:
                partition_merges[partition_idx] = partition_extractions[0]['data']
            else:
                # Use programmatic merge
                partition_merges[partition_idx] = self._merge_partition_data_programmatic(
                    partition_extractions
                )
        
        # Merge across partitions
        if len(partition_merges) == 1:
            final_merge = list(partition_merges.values())[0]
        else:
            final_merge = self._merge_across_partitions_programmatic(
                partition_merges, target_schema
            )
        
        # Ensure schema compliance
        final_merge = self._ensure_schema_compliance_programmatic(
            final_merge, target_schema
        )
        
        # Resolve conflicts programmatically
        final_merge = self._resolve_conflicts_programmatic(
            final_merge, target_schema
        )
        
        return final_merge
    
    def _create_empty_from_schema(self, schema: Dict) -> Dict:
        """Create an empty JSON structure from schema"""
        if schema.get('type') == 'object':
            result = {}
            if 'properties' in schema:
                for prop_name, prop_schema in schema['properties'].items():
                    if prop_name in schema.get('required', []):
                        result[prop_name] = self._get_default_value(prop_schema)
            return result
        elif schema.get('type') == 'array':
            return []
        else:
            return self._get_default_value(schema)
    
    def _get_default_value(self, schema: Dict) -> Any:
        """Get default value based on schema type"""
        type_defaults = {
            'string': '',
            'number': 0,
            'integer': 0,
            'boolean': False,
            'array': [],
            'object': {},
            'null': None
        }
        return type_defaults.get(schema.get('type', 'null'), None)
    
    def _group_by_partition(self, extractions: List[Dict]) -> Dict[int, List[Dict]]:
        """Group extractions by partition index"""
        partitioned = defaultdict(list)
        
        for extraction in extractions:
            partition_idx = extraction.get('partition_idx', 0)
            partitioned[partition_idx].append(extraction)
        
        return dict(partitioned)
    
    def _merge_partition_data_programmatic(self, partition_extractions: List[Dict]) -> Dict:
        """Merge data from same partition across chunks using programmatic approach"""
        
        if not partition_extractions:
            return {}
        
        if len(partition_extractions) == 1:
            return partition_extractions[0].get('data', {})
        
        # Sort by chunk index to maintain order
        sorted_extractions = sorted(
            partition_extractions, 
            key=lambda x: x.get('chunk_idx', 0)
        )
        
        # Start with first extraction as base
        merged = copy.deepcopy(sorted_extractions[0].get('data', {}))
        
        # Merge subsequent extractions
        for extraction in sorted_extractions[1:]:
            data = extraction.get('data', {})
            merged = self._deep_merge(merged, data)
        
        return merged
    
    def _merge_across_partitions_programmatic(
        self,
        partition_merges: Dict[int, Dict],
        target_schema: Dict
    ) -> Dict:
        """Merge across partitions programmatically"""
        
        if not partition_merges:
            return {}
        
        # Start with empty structure from schema
        merged = self._create_empty_from_schema(target_schema)
        
        # Merge each partition
        for partition_idx in sorted(partition_merges.keys()):
            partition_data = partition_merges[partition_idx]
            merged = self._deep_merge(merged, partition_data)
        
        return merged
    
    def _deep_merge(self, base: Any, update: Any) -> Any:
        """Deep merge two data structures"""
        
        # Determine types
        base_type = self._get_type(base)
        update_type = self._get_type(update)
        
        # If types don't match, prefer non-null
        if base_type != update_type:
            if update is not None and update != "" and update != [] and update != {}:
                return update
            return base
        
        # Use appropriate merge strategy
        strategy = self.merge_strategies.get(base_type)
        if strategy:
            return strategy(base, update)
        
        # Default: prefer non-null/non-empty
        if update is not None and update != "":
            return update
        return base
    
    def _get_type(self, value: Any) -> str:
        """Get simplified type for merge strategy"""
        if isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return 'scalar'
        else:
            return 'unknown'
    
    def _merge_scalar(self, base: Any, update: Any) -> Any:
        """Merge scalar values"""
        # Prefer non-null/non-empty values
        if update is not None and update != "":
            return update
        return base
    
    def _merge_array(self, base: List, update: List) -> List:
        """Merge arrays intelligently"""
        if not base:
            return update
        if not update:
            return base
        
        # Check if arrays contain objects
        if base and isinstance(base[0], dict):
            # Merge arrays of objects by combining unique items
            merged = list(base)
            
            for item in update:
                if item not in merged:
                    merged.append(item)
            
            return merged
        else:
            # For simple arrays, combine and deduplicate
            combined = base + update
            
            # Preserve order while removing duplicates
            seen = set()
            result = []
            for item in combined:
                # Handle unhashable types
                try:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                except TypeError:
                    # For unhashable types, just append
                    if item not in result:
                        result.append(item)
            
            return result
    
    def _merge_object(self, base: Dict, update: Dict) -> Dict:
        """Merge objects recursively"""
        merged = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in merged:
                # Recursive merge
                merged[key] = self._deep_merge(merged[key], value)
            else:
                # Add new key
                merged[key] = value
        
        return merged
    
    def _ensure_schema_compliance_programmatic(self, data: Dict, schema: Dict) -> Dict:
        """Ensure data complies with schema programmatically"""
        
        if not schema or schema.get('type') != 'object':
            return data
        
        compliant = {}
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        # Process defined properties
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                # Validate and clean the value
                compliant[prop_name] = self._validate_and_clean(
                    data[prop_name], prop_schema
                )
            elif prop_name in required:
                # Add required field with default value
                compliant[prop_name] = self._get_default_value(prop_schema)
        
        # Handle additionalProperties
        if schema.get('additionalProperties', True):
            # Add any extra properties from data
            for key, value in data.items():
                if key not in compliant:
                    compliant[key] = value
        
        return compliant
    
    def _validate_and_clean(self, value: Any, schema: Dict) -> Any:
        """Validate and clean a value according to schema"""
        expected_type = schema.get('type')
        
        if not expected_type:
            return value
        
        # Type coercion
        try:
            if expected_type == 'string':
                return str(value) if value is not None else ''
            elif expected_type == 'number':
                return float(value) if value is not None else 0.0
            elif expected_type == 'integer':
                return int(value) if value is not None else 0
            elif expected_type == 'boolean':
                return bool(value) if value is not None else False
            elif expected_type == 'array':
                if isinstance(value, list):
                    # Validate array items if schema provided
                    if 'items' in schema:
                        return [
                            self._validate_and_clean(item, schema['items'])
                            for item in value
                        ]
                    return value
                return []
            elif expected_type == 'object':
                if isinstance(value, dict):
                    # Recursively validate object
                    if 'properties' in schema:
                        return self._ensure_schema_compliance_programmatic(
                            value, schema
                        )
                    return value
                return {}
            elif expected_type == 'null':
                return None
        except (ValueError, TypeError):
            # Return default if conversion fails
            return self._get_default_value(schema)
        
        return value
    
    def _resolve_conflicts_programmatic(self, data: Dict, schema: Dict) -> Dict:
        """Resolve conflicts and redundancies programmatically"""
        
        # Remove null values for optional fields
        cleaned = {}
        required = schema.get('required', [])
        
        for key, value in data.items():
            # Keep required fields even if null
            if key in required:
                cleaned[key] = value
            # Remove optional fields if they're null or empty
            elif value is not None and value != "" and value != [] and value != {}:
                cleaned[key] = value
        
        # Ensure consistency in related fields
        cleaned = self._ensure_field_consistency(cleaned, schema)
        
        return cleaned
    
    def _ensure_field_consistency(self, data: Dict, schema: Dict) -> Dict:
        """Ensure consistency between related fields"""
        
        # Common patterns to check
        consistency_rules = [
            # If we have start_date and end_date, ensure end >= start
            ('start_date', 'end_date'),
            ('begin_date', 'finish_date'),
            ('min', 'max'),
            ('minimum', 'maximum'),
        ]
        
        for field1, field2 in consistency_rules:
            if field1 in data and field2 in data:
                try:
                    # Simple comparison for common patterns
                    if data[field1] and data[field2]:
                        if isinstance(data[field1], (int, float)) and isinstance(data[field2], (int, float)):
                            if data[field1] > data[field2]:
                                # Swap if in wrong order
                                data[field1], data[field2] = data[field2], data[field1]
                except Exception:
                    # Skip if comparison fails
                    pass
        
        return data

# Initialize global instance
merger_engine = MergerEngine()