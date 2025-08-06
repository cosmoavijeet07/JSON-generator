import json
from typing import Dict, Any, List, Optional, Tuple

class SchemaProcessor:
    def __init__(self):
        self.complexity_threshold = 10
    
    def analyze_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema complexity and structure"""
        analysis = {
            "total_fields": self._count_fields(schema),
            "max_depth": self._get_max_depth(schema),
            "has_arrays": self._has_arrays(schema),
            "has_nested_objects": self._has_nested_objects(schema),
            "required_fields": self._get_required_fields(schema),
            "optional_fields": self._get_optional_fields(schema),
            "field_types": self._analyze_field_types(schema),
            "complexity_score": 0,
            "needs_partitioning": False,
            "partition_strategy": None
        }
        
        # Calculate complexity score
        analysis["complexity_score"] = self._calculate_complexity(analysis)
        analysis["needs_partitioning"] = analysis["complexity_score"] > self.complexity_threshold
        
        if analysis["needs_partitioning"]:
            analysis["partition_strategy"] = self._suggest_partition_strategy(schema, analysis)
        
        return analysis
    
    def partition_schema(self, schema: Dict[str, Any], strategy: str = "auto") -> List[Dict[str, Any]]:
        """Partition schema into manageable parts"""
        if strategy == "auto":
            analysis = self.analyze_schema(schema)
            strategy = analysis.get("partition_strategy", "by_depth")
        
        if strategy == "by_depth":
            return self._partition_by_depth(schema)
        elif strategy == "by_field_groups":
            return self._partition_by_field_groups(schema)
        elif strategy == "by_required":
            return self._partition_by_required(schema)
        else:
            return [schema]  # No partitioning
    
    def _partition_by_depth(self, schema: Dict[str, Any], max_depth: int = 2) -> List[Dict[str, Any]]:
        """Partition schema by nesting depth"""
        partitions = []
        
        # First partition: top-level fields only
        top_level = {
            "type": schema.get("type", "object"),
            "properties": {},
            "required": []
        }
        
        nested_schemas = []
        
        if "properties" in schema:
            for key, value in schema["properties"].items():
                if isinstance(value, dict) and value.get("type") == "object":
                    # Save nested object for separate partition
                    nested_schemas.append({
                        "parent_key": key,
                        "schema": value
                    })
                    # Reference in top level
                    top_level["properties"][key] = {"type": "object", "$ref": f"#{key}"}
                else:
                    top_level["properties"][key] = value
            
            # Handle required fields
            if "required" in schema:
                top_level["required"] = [r for r in schema["required"] 
                                        if r in top_level["properties"]]
        
        partitions.append(top_level)
        
        # Add nested schemas as separate partitions
        for nested in nested_schemas:
            partitions.append({
                "partition_type": "nested",
                "parent_key": nested["parent_key"],
                "schema": nested["schema"]
            })
        
        return partitions
    
    def _partition_by_field_groups(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Partition schema by logical field groups"""
        if "properties" not in schema:
            return [schema]
        
        # Group fields by similarity or prefix
        groups = {}
        
        for key, value in schema["properties"].items():
            # Simple grouping by prefix
            prefix = key.split('_')[0] if '_' in key else key[:3]
            
            if prefix not in groups:
                groups[prefix] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            groups[prefix]["properties"][key] = value
            
            if "required" in schema and key in schema["required"]:
                groups[prefix]["required"].append(key)
        
        return list(groups.values())
    
    def _partition_by_required(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Partition schema by required vs optional fields"""
        if "properties" not in schema:
            return [schema]
        
        required_fields = schema.get("required", [])
        
        required_partition = {
            "type": "object",
            "properties": {k: v for k, v in schema["properties"].items() 
                         if k in required_fields},
            "required": required_fields
        }
        
        optional_partition = {
            "type": "object",
            "properties": {k: v for k, v in schema["properties"].items() 
                         if k not in required_fields},
            "required": []
        }
        
        partitions = []
        if required_partition["properties"]:
            partitions.append(required_partition)
        if optional_partition["properties"]:
            partitions.append(optional_partition)
        
        return partitions if partitions else [schema]
    
    def _count_fields(self, schema: Dict[str, Any]) -> int:
        """Count total fields in schema"""
        count = 0
        
        if isinstance(schema, dict):
            if "properties" in schema:
                count += len(schema["properties"])
                for prop in schema["properties"].values():
                    count += self._count_fields(prop)
            elif "items" in schema:
                count += self._count_fields(schema["items"])
        
        return count
    
    def _get_max_depth(self, schema: Dict[str, Any], current_depth: int = 0) -> int:
        """Get maximum nesting depth"""
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
    
    def _has_arrays(self, schema: Dict[str, Any]) -> bool:
        """Check if schema contains arrays"""
        if isinstance(schema, dict):
            if schema.get("type") == "array":
                return True
            
            if "properties" in schema:
                for prop in schema["properties"].values():
                    if self._has_arrays(prop):
                        return True
        
        return False
    
    def _has_nested_objects(self, schema: Dict[str, Any]) -> bool:
        """Check if schema contains nested objects"""
        if isinstance(schema, dict) and "properties" in schema:
            for prop in schema["properties"].values():
                if isinstance(prop, dict) and prop.get("type") == "object":
                    return True
        
        return False
    
    def _get_required_fields(self, schema: Dict[str, Any]) -> List[str]:
        """Get all required fields"""
        required = []
        
        if isinstance(schema, dict):
            if "required" in schema:
                required.extend(schema["required"])
            
            if "properties" in schema:
                for key, prop in schema["properties"].items():
                    nested_required = self._get_required_fields(prop)
                    required.extend([f"{key}.{field}" for field in nested_required])
        
        return required
    
    def _get_optional_fields(self, schema: Dict[str, Any]) -> List[str]:
        """Get all optional fields"""
        all_fields = self._get_all_fields(schema)
        required_fields = self._get_required_fields(schema)
        return [f for f in all_fields if f not in required_fields]
    
    def _get_all_fields(self, schema: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all fields in schema"""
        fields = []
        
        if isinstance(schema, dict) and "properties" in schema:
            for key, prop in schema["properties"].items():
                field_name = f"{prefix}.{key}" if prefix else key
                fields.append(field_name)
                
                # Recursively get nested fields
                nested_fields = self._get_all_fields(prop, field_name)
                fields.extend(nested_fields)
        
        return fields
    
    def _analyze_field_types(self, schema: Dict[str, Any]) -> Dict[str, int]:
        """Analyze distribution of field types"""
        type_counts = {
            "string": 0,
            "number": 0,
            "integer": 0,
            "boolean": 0,
            "array": 0,
            "object": 0,
            "null": 0
        }
        
        self._count_types(schema, type_counts)
        
        return type_counts
    
    def _count_types(self, schema: Dict[str, Any], type_counts: Dict[str, int]):
        """Recursively count field types"""
        if isinstance(schema, dict):
            field_type = schema.get("type")
            
            if field_type:
                if isinstance(field_type, list):
                    for t in field_type:
                        if t in type_counts:
                            type_counts[t] += 1
                elif field_type in type_counts:
                    type_counts[field_type] += 1
            
            if "properties" in schema:
                for prop in schema["properties"].values():
                    self._count_types(prop, type_counts)
            
            if "items" in schema:
                self._count_types(schema["items"], type_counts)
    
    def _calculate_complexity(self, analysis: Dict[str, Any]) -> float:
        """Calculate complexity score"""
        score = 0
        
        # Depth contributes significantly
        score += analysis["max_depth"] * 3
        
        # Number of fields
        score += analysis["total_fields"] * 0.5
        
        # Arrays and nested objects add complexity
        if analysis["has_arrays"]:
            score += 2
        if analysis["has_nested_objects"]:
            score += 3
        
        # Field type diversity
        type_diversity = len([t for t, c in analysis["field_types"].items() if c > 0])
        score += type_diversity * 0.5
        
        return score
    
    def _suggest_partition_strategy(self, schema: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Suggest best partition strategy"""
        if analysis["max_depth"] > 3:
            return "by_depth"
        elif analysis["total_fields"] > 20:
            return "by_field_groups"
        elif len(analysis["required_fields"]) > 10:
            return "by_required"
        else:
            return "none"
