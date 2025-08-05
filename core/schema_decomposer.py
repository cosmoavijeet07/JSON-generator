import json
from typing import Dict, List, Any

class SchemaDecomposer:
    def __init__(self):
        pass
    
    def decompose_schema(self, schema: Dict) -> List[Dict]:
        """Break down complex schema into manageable sub-schemas"""
        sub_schemas = []
        
        if schema.get('type') == 'object' and 'properties' in schema:
            # Create sub-schemas for complex nested objects
            for prop_name, prop_schema in schema['properties'].items():
                if self._is_complex_property(prop_schema):
                    sub_schema = {
                        'title': f"{schema.get('title', 'Root')}_{prop_name}",
                        'type': 'object',
                        'properties': {prop_name: prop_schema},
                        'required': [prop_name] if prop_name in schema.get('required', []) else [],
                        '_parent_schema': schema.get('title', 'Root'),
                        '_property_name': prop_name
                    }
                    sub_schemas.append(sub_schema)
        
        # If no complex properties found, return the original schema
        if not sub_schemas:
            sub_schemas.append(schema)
        
        return sub_schemas
    
    def create_priority_order(self, schema: Dict) -> List[str]:
        """Determine extraction priority based on field importance"""
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        priority_order = []
        
        # High priority: required fields
        for field in required:
            if field in properties:
                priority_order.append(field)
        
        # Medium priority: simple optional fields
        for field, prop in properties.items():
            if field not in required and not self._is_complex_property(prop):
                priority_order.append(field)
        
        # Low priority: complex optional fields
        for field, prop in properties.items():
            if field not in required and self._is_complex_property(prop):
                priority_order.append(field)
        
        return priority_order
    
    def _is_complex_property(self, prop_schema: Dict) -> bool:
        """Determine if a property is complex enough to warrant separate extraction"""
        if prop_schema.get('type') == 'object':
            return True
        elif prop_schema.get('type') == 'array':
            items = prop_schema.get('items', {})
            return items.get('type') == 'object'
        return False
