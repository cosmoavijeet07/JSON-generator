import json
import re
from typing import List, Dict, Any, Optional, Callable
from .llm_interface import call_llm
from .logger_service import log
from .token_estimator import estimate_tokens
from jsonschema import Draft7Validator, exceptions as jsonschema_exceptions

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

class SchemaAnalyzer:
    """Handles JSON schema analysis, complexity calculation, and intelligent partitioning"""
    
    def __init__(self):
        self.max_partition_size = 50  # Max fields per partition
        self.generated_functions = {}  # Cache for generated functions
    
    def calculate_nesting_level(self, schema: Dict, current_level: int = 0) -> int:
        """Calculate the maximum nesting level of a JSON schema"""
        max_level = current_level
        
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == 'properties' and isinstance(value, dict):
                    for prop_key, prop_value in value.items():
                        level = self.calculate_nesting_level(prop_value, current_level + 1)
                        max_level = max(max_level, level)
                elif key == 'items':
                    level = self.calculate_nesting_level(value, current_level + 1)
                    max_level = max(max_level, level)
                elif isinstance(value, dict):
                    level = self.calculate_nesting_level(value, current_level)
                    max_level = max(max_level, level)
        
        return max_level
    
    def count_fields(self, schema: Dict) -> int:
        """Count total number of fields in a JSON schema"""
        count = 0
        
        if isinstance(schema, dict):
            if 'properties' in schema:
                properties = schema['properties']
                if isinstance(properties, dict):
                    count += len(properties)
                    for prop_value in properties.values():
                        count += self.count_fields(prop_value)
            
            if 'items' in schema:
                count += self.count_fields(schema['items'])
        
        return count
    
    def get_required_fields(self, schema: Dict, path: str = "") -> List[str]:
        """Get all required fields with their paths"""
        required_fields = []
        
        if isinstance(schema, dict):
            # Get required fields at current level
            if 'required' in schema and isinstance(schema['required'], list):
                for field in schema['required']:
                    field_path = f"{path}.{field}" if path else field
                    required_fields.append(field_path)
            
            # Recursively check properties
            if 'properties' in schema and isinstance(schema['properties'], dict):
                for prop_name, prop_schema in schema['properties'].items():
                    prop_path = f"{path}.{prop_name}" if path else prop_name
                    required_fields.extend(self.get_required_fields(prop_schema, prop_path))
            
            # Check items for arrays
            if 'items' in schema:
                items_path = f"{path}[]" if path else "[]"
                required_fields.extend(self.get_required_fields(schema['items'], items_path))
        
        return required_fields
    
    def calculate_complexity_score(self, tokens: int, nesting: int, fields: int) -> float:
        """Calculate a complexity score for the schema"""
        # Weighted formula for complexity
        token_weight = 0.3
        nesting_weight = 0.4
        field_weight = 0.3
        
        # Normalize values (rough estimates)
        normalized_tokens = min(tokens / 10000, 1.0)  # Cap at 10k tokens
        normalized_nesting = min(nesting / 10, 1.0)   # Cap at 10 levels
        normalized_fields = min(fields / 100, 1.0)    # Cap at 100 fields
        
        score = (
            token_weight * normalized_tokens +
            nesting_weight * normalized_nesting +
            field_weight * normalized_fields
        ) * 100
        
        return score
    
    def analyze_complexity(self, schema: Dict, model: str) -> Dict[str, Any]:
        """Analyze schema complexity locally without LLM"""
        
        # Local analysis
        nesting_level = self.calculate_nesting_level(schema)
        field_count = self.count_fields(schema)
        required_fields = self.get_required_fields(schema)
        
        # Determine complexity
        if nesting_level <= 2 and field_count <= 10:
            complexity = "simple"
        elif nesting_level <= 4 and field_count <= 30:
            complexity = "moderate"
        elif nesting_level <= 6 and field_count <= 60:
            complexity = "complex"
        else:
            complexity = "very_complex"
        
        # Find logical groups (simplified local analysis)
        logical_groups = self._find_logical_groups(schema)
        
        # Find dependencies
        dependencies = self._find_dependencies(schema)
        
        # Generate partition suggestions
        partition_suggestions = self._generate_partition_suggestions(schema)
        
        return {
            "complexity": complexity,
            "logical_groups": logical_groups,
            "dependencies": dependencies,
            "partition_suggestions": partition_suggestions,
            "must_stay_together": [],
            "statistics": {
                "nesting_level": nesting_level,
                "field_count": field_count,
                "required_count": len(required_fields)
            }
        }
    
    def _find_logical_groups(self, schema: Dict) -> List[Dict]:
        """Find logical groups of fields locally"""
        groups = []
        
        if 'properties' in schema:
            # Group by common prefixes
            prefix_groups = {}
            for field_name in schema['properties'].keys():
                # Extract prefix (e.g., "user_" from "user_name")
                if '_' in field_name:
                    prefix = field_name.split('_')[0]
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(field_name)
            
            # Create groups from prefixes
            for prefix, fields in prefix_groups.items():
                if len(fields) > 1:
                    groups.append({
                        "name": f"{prefix}_fields",
                        "fields": fields,
                        "reason": f"Common prefix: {prefix}"
                    })
            
            # Group nested objects
            for field_name, field_schema in schema['properties'].items():
                if field_schema.get('type') == 'object' and 'properties' in field_schema:
                    groups.append({
                        "name": f"{field_name}_object",
                        "fields": [field_name],
                        "reason": "Nested object structure"
                    })
        
        return groups
    
    def _find_dependencies(self, schema: Dict) -> List[Dict]:
        """Find field dependencies locally"""
        dependencies = []
        
        # Check for conditional requirements
        if 'dependencies' in schema:
            for field, deps in schema['dependencies'].items():
                if isinstance(deps, list):
                    dependencies.append({
                        "field": field,
                        "depends_on": deps,
                        "type": "required"
                    })
        
        # Check for if/then/else conditions
        if 'if' in schema and 'then' in schema:
            dependencies.append({
                "field": "conditional_fields",
                "depends_on": ["if_condition"],
                "type": "conditional"
            })
        
        return dependencies
    
    def _generate_partition_suggestions(self, schema: Dict) -> List[Dict]:
        """Generate partition suggestions locally"""
        suggestions = []
        
        if 'properties' not in schema:
            return suggestions
        
        properties = schema['properties']
        prop_count = len(properties)
        
        if prop_count <= self.max_partition_size:
            # Single partition
            suggestions.append({
                "name": "main",
                "path": "",
                "estimated_complexity": "low"
            })
        else:
            # Multiple partitions
            partition_count = (prop_count // self.max_partition_size) + 1
            props_list = list(properties.keys())
            
            for i in range(partition_count):
                start_idx = i * self.max_partition_size
                end_idx = min((i + 1) * self.max_partition_size, prop_count)
                
                suggestions.append({
                    "name": f"partition_{i+1}",
                    "path": f"properties[{start_idx}:{end_idx}]",
                    "estimated_complexity": "medium" if end_idx - start_idx > 20 else "low"
                })
        
        return suggestions
    
    def generate_partitioning_strategy(self, schema: Dict, complexity_analysis: Dict, model: str) -> str:
        """Generate Python code for schema partitioning"""
        
        # Create a focused prompt for code generation
        prompt = f"""Generate a Python function to partition a JSON schema intelligently.

Schema Statistics:
- Nesting Level: {complexity_analysis['statistics']['nesting_level']}
- Field Count: {complexity_analysis['statistics']['field_count']}
- Complexity: {complexity_analysis['complexity']}

Requirements:
- Function signature: def partition_schema(schema: Dict) -> List[Dict]
- Each partition should be a valid sub-schema
- Maximum {self.max_partition_size} fields per partition
- Preserve required field constraints
- Keep nested objects intact when possible

Return ONLY executable Python code:
```python
def partition_schema(schema: Dict) -> List[Dict]:
    # Your implementation
    pass
```"""
        
        # Get code from LLM
        response = call_llm(prompt, model)
        
        # Extract and validate code
        code = self._extract_code_from_response(response)
        code = self._validate_and_fix_code(code)
        
        # Store for potential reuse
        self.generated_functions['partition_schema'] = code
        
        return code
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code block
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Try to find function definition
        func_match = re.search(r'(def partition_schema.*?)(?=\n\n|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1)
        
        # Return default if no code found
        return self._get_default_partitioning_code()
    
    def _validate_and_fix_code(self, code: str) -> str:
        """Validate and fix generated code"""
        try:
            # Check if code can be compiled
            compile(code, '<string>', 'exec')
            
            # Ensure required imports
            if 'Dict' in code and 'from typing import' not in code:
                code = "from typing import Dict, List\n" + code
            
            return code
        
        except SyntaxError:
            return self._get_default_partitioning_code()
    
    def _get_default_partitioning_code(self) -> str:
        """Get default partitioning implementation"""
        return '''
from typing import Dict, List

def partition_schema(schema: Dict) -> List[Dict]:
    """Intelligent schema partitioning"""
    if not schema or 'properties' not in schema:
        return [schema] if schema else []
    
    properties = schema.get('properties', {})
    max_fields = 30  # Maximum fields per partition
    
    # If small enough, return as single partition
    if len(properties) <= max_fields:
        return [schema]
    
    # Partition properties
    partitions = []
    props_list = list(properties.items())
    
    for i in range(0, len(props_list), max_fields):
        chunk_props = dict(props_list[i:i+max_fields])
        
        # Create partition schema
        partition = {
            'type': schema.get('type', 'object'),
            'properties': chunk_props
        }
        
        # Add required fields for this partition
        if 'required' in schema:
            partition_required = [
                field for field in schema['required']
                if field in chunk_props
            ]
            if partition_required:
                partition['required'] = partition_required
        
        # Copy other schema properties
        for key in ['title', 'description', 'additionalProperties']:
            if key in schema:
                partition[key] = schema[key]
        
        partitions.append(partition)
    
    return partitions
'''
    
    def execute_partitioning(self, schema: Dict, partitioning_code: str) -> List[Dict]:
        """Execute the partitioning code on the schema"""
        
        try:
            # Create safe execution environment
            exec_namespace = {
                '__builtins__': {
                    'len': len,
                    'dict': dict,
                    'list': list,
                    'range': range,
                    'min': min,
                    'max': max,
                    'isinstance': isinstance
                },
                'Dict': Dict,
                'List': List
            }
            
            # Execute the code
            exec(partitioning_code, exec_namespace)
            
            # Get the partition function
            partition_function = exec_namespace.get('partition_schema')
            
            if partition_function and callable(partition_function):
                partitions = partition_function(schema)
                
                # Validate partitions
                if isinstance(partitions, list):
                    return partitions if partitions else [schema]
            
            return [schema]
        
        except Exception as e:
            log("session", "partitioning_error", f"Error executing partitioning code: {e}", "ERROR")
            return [schema]
    
    def get_partition_metrics(self, schema: Dict, partitioning_code: str) -> Dict[str, Any]:
        """Get metrics about the partitioning strategy"""
        
        try:
            partitions = self.execute_partitioning(schema, partitioning_code)
            
            if not partitions:
                return {
                    'partition_count': 0,
                    'avg_fields': 0,
                    'max_depth': 0,
                    'balanced': False
                }
            
            field_counts = [self.count_fields(p) for p in partitions]
            depths = [self.calculate_nesting_level(p) for p in partitions]
            
            # Check if partitions are balanced
            balanced = True
            if len(field_counts) > 1:
                avg_fields = sum(field_counts) / len(field_counts)
                for count in field_counts:
                    if abs(count - avg_fields) > avg_fields * 0.5:
                        balanced = False
                        break
            
            return {
                'partition_count': len(partitions),
                'avg_fields': sum(field_counts) / len(field_counts) if field_counts else 0,
                'max_depth': max(depths) if depths else 0,
                'balanced': balanced
            }
        
        except Exception as e:
            log("session", "metrics_error", f"Error getting partition metrics: {e}", "ERROR")
            return {
                'partition_count': 'Error',
                'avg_fields': 0,
                'max_depth': 0,
                'balanced': False
            }

# Initialize global instance
schema_analyzer = SchemaAnalyzer()

