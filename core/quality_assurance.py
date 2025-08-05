import json
from typing import Dict, List, Any, Tuple
import re
from datetime import datetime

class QualityAssurance:
    def __init__(self):
        pass
    
    def validate_extraction(self, result: Dict, schema: Dict, original_text: str) -> Tuple[bool, List[str]]:
        """Comprehensive validation of extracted data"""
        issues = []
        
        # Schema compliance check
        schema_issues = self._check_schema_compliance(result, schema)
        issues.extend(schema_issues)
        
        # Logical consistency check
        logic_issues = self._check_logical_consistency(result)
        issues.extend(logic_issues)
        
        # Completeness check
        completeness_issues = self._check_completeness(result, schema, original_text)
        issues.extend(completeness_issues)
        
        # Data quality check
        quality_issues = self._check_data_quality(result)
        issues.extend(quality_issues)
        
        return len(issues) == 0, issues
    
    def calculate_confidence_score(self, result: Dict, original_text: str) -> float:
        """Calculate confidence score for the extraction"""
        score = 1.0
        
        # Check if key information is present
        if not result:
            return 0.0
        
        # Penalize for empty or null values
        total_fields = len(result)
        empty_fields = sum(1 for v in result.values() if v is None or v == "" or v == [])
        if total_fields > 0:
            score -= (empty_fields / total_fields) * 0.3
        
        # Reward for finding specific patterns in original text
        for key, value in result.items():
            if isinstance(value, str) and value and value.lower() in original_text.lower():
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_schema_compliance(self, result: Dict, schema: Dict) -> List[str]:
        """Check if result complies with schema requirements"""
        issues = []
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        # Check required fields
        for field in required:
            if field not in result or result[field] is None:
                issues.append(f"Missing required field: {field}")
        
        # Check field types
        for field, value in result.items():
            if field in properties:
                expected_type = properties[field].get('type')
                if not self._check_type_match(value, expected_type):
                    issues.append(f"Type mismatch for field {field}: expected {expected_type}")
        
        return issues
    
    def _check_logical_consistency(self, result: Dict) -> List[str]:
        """Check logical consistency of extracted data"""
        issues = []
        
        # Date consistency checks
        date_fields = [k for k, v in result.items() if 'date' in k.lower() or 'time' in k.lower()]
        if len(date_fields) >= 2:
            dates = []
            for field in date_fields:
                date_str = result.get(field)
                if date_str and isinstance(date_str, str):
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        dates.append((field, date_obj))
                    except:
                        pass
            
            # Check if dates are in logical order
            if len(dates) >= 2:
                dates.sort(key=lambda x: x[1])
                # Add specific date logic checks here
        
        # Numerical consistency checks
        numeric_fields = [k for k, v in result.items() if isinstance(v, (int, float))]
        if 'total' in result and 'subtotal' in result:
            if result['total'] < result['subtotal']:
                issues.append("Total cannot be less than subtotal")
        
        return issues
    
    def _check_completeness(self, result: Dict, schema: Dict, original_text: str) -> List[str]:
        """Check if extraction appears complete"""
        issues = []
        
        # Check if we're missing obvious information
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field not in result or not result[field]:
                # Check if the information might be available in text
                if self._likely_contains_info(original_text, field):
                    issues.append(f"Potentially missing information for field: {field}")
        
        return issues
    
    def _check_data_quality(self, result: Dict) -> List[str]:
        """Check quality of extracted data"""
        issues = []
        
        for field, value in result.items():
            if isinstance(value, str):
                # Check for incomplete extractions
                if value.endswith('...') or value.startswith('...'):
                    issues.append(f"Incomplete extraction for field: {field}")
                
                # Check for extraction artifacts
                if any(artifact in value.lower() for artifact in ['extract', 'found', 'information']):
                    issues.append(f"Extraction artifact detected in field: {field}")
        
        return issues
    
    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        return True
    
    def _likely_contains_info(self, text: str, field: str) -> bool:
        """Heuristic to check if text likely contains information for a field"""
        field_keywords = {
            'name': ['name', 'called', 'known as'],
            'email': ['email', '@', 'contact'],
            'phone': ['phone', 'tel', 'call'],
            'address': ['address', 'street', 'city'],
            'date': ['date', 'when', 'time'],
            'amount': ['amount', 'cost', 'price', '$'],
        }
        
        keywords = field_keywords.get(field.lower(), [field.lower()])
        return any(keyword in text.lower() for keyword in keywords)
