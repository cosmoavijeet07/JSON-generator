import json
import os
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

class FileManager:
    @staticmethod
    def save_json(data: dict, filename: str, directory: str = "temp") -> str:
        """Save JSON data to file"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @staticmethod
    def load_json(filepath: str) -> dict:
        """Load JSON from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def validate_uploaded_file(uploaded_file, allowed_extensions: List[str], max_size: int) -> tuple[bool, str]:
        """Validate uploaded file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in allowed_extensions:
            return False, f"File type {file_extension} not allowed. Allowed types: {', '.join(allowed_extensions)}"
        
        # Check file size
        if uploaded_file.size > max_size:
            return False, f"File size ({uploaded_file.size / 1024 / 1024:.2f}MB) exceeds maximum ({max_size / 1024 / 1024:.0f}MB)"
        
        return True, "File is valid"

class LogManager:
    @staticmethod
    def create_log_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert logs to pandas DataFrame"""
        if not logs:
            return pd.DataFrame(columns=['Timestamp', 'Step', 'Status', 'Details'])
        
        df = pd.DataFrame(logs)
        df = df.rename(columns={
            'timestamp': 'Timestamp',
            'step': 'Step', 
            'status': 'Status',
            'details': 'Details'
        })
        
        return df[['Timestamp', 'Step', 'Status', 'Details']]
    
    @staticmethod
    def save_logs(logs: List[Dict[str, Any]], filename: str = None) -> str:
        """Save logs to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"extraction_logs_{timestamp}.json"
        
        os.makedirs('logs', exist_ok=True)
        filepath = os.path.join('logs', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        return filepath

class SchemaValidator:
    @staticmethod
    def validate_json_schema(schema_data: dict) -> tuple[bool, str]:
        """Validate if the provided data is a valid JSON schema"""
        try:
            # Basic validation - check if it has the typical schema structure
            if not isinstance(schema_data, dict):
                return False, "Schema must be a JSON object"
            
            # Check for basic schema properties - be more flexible
            # A schema can be valid without 'type' if it has other valid schema keywords
            valid_root_keywords = {
                'type', 'properties', 'items', 'additionalProperties', 
                'required', 'enum', 'const', 'anyOf', 'oneOf', 'allOf',
                '$ref', '$schema', '$id', 'title', 'description', 
                'default', 'examples', 'minimum', 'maximum', 'pattern',
                'minLength', 'maxLength', 'minItems', 'maxItems'
            }
            
            # Check if schema has at least one valid keyword
            if not any(keyword in schema_data for keyword in valid_root_keywords):
                return False, "Schema must contain at least one valid JSON Schema keyword"
            
            # If it's an object type, validate properties structure
            if schema_data.get('type') == 'object' and 'properties' in schema_data:
                properties = schema_data['properties']
                if not isinstance(properties, dict):
                    return False, "'properties' must be an object"
                
                # Validate each property - be more flexible
                for prop_name, prop_schema in properties.items():
                    if not isinstance(prop_schema, dict):
                        return False, f"Property '{prop_name}' schema must be an object"
                    
                    # A property is valid if it has any valid schema keywords
                    # Not every property needs a 'type' - it could have $ref, anyOf, etc.
                    valid_prop_keywords = {
                        'type', 'enum', 'const', 'anyOf', 'oneOf', 'allOf',
                        '$ref', 'properties', 'items', 'additionalProperties',
                        'title', 'description', 'default', 'examples',
                        'minimum', 'maximum', 'pattern', 'minLength', 'maxLength',
                        'minItems', 'maxItems', 'format'
                    }
                    
                    if not any(keyword in prop_schema for keyword in valid_prop_keywords):
                        return False, f"Property '{prop_name}' must contain at least one valid schema keyword"
            
            # Additional validation for array schemas
            if schema_data.get('type') == 'array':
                if 'items' in schema_data and not isinstance(schema_data['items'], (dict, list)):
                    return False, "'items' must be an object or array"
            
            return True, "Valid JSON schema"
            
        except Exception as e:
            return False, f"Schema validation error: {str(e)}"
    
    @staticmethod
    def get_schema_complexity(schema: dict) -> str:
        """Analyze schema complexity"""
        complexity_score = 0
        
        def analyze_properties(props: dict, depth: int = 0):
            nonlocal complexity_score
            complexity_score += len(props) * (depth + 1)
            
            for prop_schema in props.values():
                prop_type = prop_schema.get('type')
                
                # Handle different schema patterns
                if prop_type == 'object' and 'properties' in prop_schema:
                    analyze_properties(prop_schema['properties'], depth + 1)
                elif prop_type == 'array':
                    complexity_score += 2
                    items_schema = prop_schema.get('items', {})
                    if isinstance(items_schema, dict) and items_schema.get('type') == 'object':
                        if 'properties' in items_schema:
                            analyze_properties(items_schema['properties'], depth + 1)
                
                # Handle schema composition (anyOf, oneOf, allOf)
                for composition_key in ['anyOf', 'oneOf', 'allOf']:
                    if composition_key in prop_schema:
                        complexity_score += 3  # Composition adds complexity
                        for sub_schema in prop_schema[composition_key]:
                            if isinstance(sub_schema, dict) and 'properties' in sub_schema:
                                analyze_properties(sub_schema['properties'], depth + 1)
                
                # Handle $ref (references add some complexity)
                if '$ref' in prop_schema:
                    complexity_score += 2
        
        if 'properties' in schema:
            analyze_properties(schema['properties'])
        
        # Handle root-level composition
        for composition_key in ['anyOf', 'oneOf', 'allOf']:
            if composition_key in schema:
                complexity_score += 5
        
        if complexity_score < 10:
            return "low"
        elif complexity_score < 25:
            return "medium"
        else:
            return "high"

class UIHelper:
    @staticmethod
    def render_processing_status(step: str, status: str):
        """Render processing status in a consistent format"""
        status_icons = {
            'running': '🔄',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        icon = status_icons.get(status, 'ℹ️')
        
        if status == 'running':
            st.info(f"{icon} {step}...")
        elif status == 'success':
            st.success(f"{icon} {step} completed")
        elif status == 'error':
            st.error(f"{icon} {step} failed")
        elif status == 'warning':
            st.warning(f"{icon} {step}")
        else:
            st.info(f"{icon} {step}")
    
    @staticmethod
    def render_usage_metrics(usage_data: Dict[str, Any]):
        """Render token usage and cost metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Input Tokens",
                f"{usage_data.get('total_input_tokens', 0):,}",
                help="Total tokens sent to the model"
            )
        
        with col2:
            st.metric(
                "Output Tokens", 
                f"{usage_data.get('total_output_tokens', 0):,}",
                help="Total tokens generated by the model"
            )
        
        with col3:
            st.metric(
                "Total Tokens",
                f"{usage_data.get('total_tokens', 0):,}",
                help="Input + Output tokens"
            )
        
        with col4:
            st.metric(
                "Estimated Cost",
                f"${usage_data.get('total_cost', 0.0):.4f}",
                help="Estimated API cost for this extraction"
            )
    
    @staticmethod
    def create_download_buttons(json_data: dict, logs: List[Dict], prefix: str = "extraction"):
        """Create download buttons for JSON and logs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{prefix}_result_{timestamp}.json",
                mime="application/json",
                help="Download the extracted JSON data"
            )
        
        with col2:
            logs_str = json.dumps(logs, indent=2, ensure_ascii=False)
            st.download_button(
                label="📋 Download Logs",
                data=logs_str,
                file_name=f"{prefix}_logs_{timestamp}.json",
                mime="application/json",
                help="Download the processing logs"
            )
