import json
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from models import ModelInterface
from text_processor import TextProcessor
import time
from datetime import datetime

class JSONExtractor:
    def __init__(self, model_interface: ModelInterface):
        self.model_interface = model_interface
        self.text_processor = TextProcessor()
        self.extraction_logs = []
    
    def log_step(self, step: str, status: str, details: str = ""):
        """Log extraction steps"""
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'step': step,
            'status': status,
            'details': details
        }
        self.extraction_logs.append(log_entry)
        
        # Display in Streamlit
        if status == 'success':
            st.success(f"✅ {step}: {details}")
        elif status == 'error':
            st.error(f"❌ {step}: {details}")
        elif status == 'warning':
            st.warning(f"⚠️ {step}: {details}")
        else:
            st.info(f"ℹ️ {step}: {details}")
    
    def create_extraction_prompt(self, schema: dict, text_chunk: str, context: str = "") -> str:
        """Create extraction prompt for LLM"""
        
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""
Extract structured data from the following text and format it according to the provided JSON schema.

JSON SCHEMA TO FOLLOW:
{schema_str}

EXTRACTION RULES:
1. Extract ALL relevant information that matches the schema fields
2. If a field is not present in the text, use null for optional fields or appropriate default values
3. Maintain data types as specified in the schema
4. For arrays, extract all matching items found in the text
5. Ensure the output is valid JSON that validates against the schema
6. Be precise and avoid hallucination - only extract information actually present in the text

{f"CONTEXT: {context}" if context else ""}

TEXT TO PROCESS:
{text_chunk}

Return ONLY the extracted JSON data, no additional text or explanation.
"""
        return prompt
    
    def extract_from_single_chunk(self, 
                                 schema: dict, 
                                 chunk: dict, 
                                 model: str,
                                 context: str = "") -> Dict[str, Any]:
        """Extract JSON from a single text chunk"""
        
        try:
            prompt = self.create_extraction_prompt(schema, chunk['text'], context)
            
            result = self.model_interface.call_model(
                prompt=prompt,
                model=model,
                json_mode=True
            )
            
            if result['success']:
                is_valid, parsed_json, message = self.model_interface.validate_json_output(result['content'])
                
                return {
                    'chunk_id': chunk['chunk_id'],
                    'success': is_valid,
                    'data': parsed_json if is_valid else None,
                    'raw_response': result['content'],
                    'usage': result['usage'],
                    'validation_message': message,
                    'processing_time': time.time()
                }
            else:
                return {
                    'chunk_id': chunk['chunk_id'],
                    'success': False,
                    'error': result['error'],
                    'attempt': result.get('attempt', 1)
                }
                
        except Exception as e:
            return {
                'chunk_id': chunk['chunk_id'],
                'success': False,
                'error': str(e)
            }
    
    def parallel_extraction(self, 
                          schema: dict, 
                          chunks: List[dict], 
                          model: str,
                          max_workers: int = 3) -> List[Dict[str, Any]]:
        """Process multiple chunks in parallel"""
        
        self.log_step("Parallel Processing", "info", f"Processing {len(chunks)} chunks with {max_workers} workers")
        
        results = []
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.extract_from_single_chunk, schema, chunk, model): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                result = future.result()
                results.append(result)
                
                # Update progress
                progress = (i + 1) / len(chunks)
                st.progress(progress)
                
                if result['success']:
                    self.log_step(f"Chunk {result['chunk_id']}", "success", "Extracted successfully")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    self.log_step(f"Chunk {result['chunk_id']}", "error", error_msg)
        
        return sorted(results, key=lambda x: x['chunk_id'])
    
    def merge_extraction_results(self, results: List[Dict[str, Any]], schema: dict) -> Dict[str, Any]:
        """Merge results from multiple chunks using MapReduce approach"""
        
        self.log_step("Result Merging", "info", "Starting MapReduce merge process")
        
        successful_results = [r for r in results if r['success'] and r['data']]
        
        if not successful_results:
            self.log_step("Result Merging", "error", "No successful extractions to merge")
            return {'success': False, 'error': 'No successful extractions'}
        
        # Initialize merged result based on schema structure
        merged_data = self._initialize_merged_structure(schema)
        
        # Merge each successful result
        for result in successful_results:
            try:
                merged_data = self._merge_single_result(merged_data, result['data'], schema)
            except Exception as e:
                self.log_step("Result Merging", "warning", f"Failed to merge chunk {result['chunk_id']}: {str(e)}")
        
        # Post-process merged data
        merged_data = self._post_process_merged_data(merged_data, schema)
        
        # Calculate total usage
        total_usage = self._calculate_total_usage(results)
        
        self.log_step("Result Merging", "success", f"Successfully merged {len(successful_results)} chunks")
        
        return {
            'success': True,
            'data': merged_data,
            'chunk_results': results,
            'total_usage': total_usage,
            'merge_summary': {
                'total_chunks': len(results),
                'successful_chunks': len(successful_results),
                'failed_chunks': len(results) - len(successful_results)
            }
        }
    
    def _initialize_merged_structure(self, schema: dict) -> dict:
        """Initialize merged data structure based on schema"""
        merged = {}
        
        if 'properties' in schema:
            for key, prop_schema in schema['properties'].items():
                if prop_schema.get('type') == 'array':
                    merged[key] = []
                elif prop_schema.get('type') == 'object':
                    merged[key] = self._initialize_merged_structure(prop_schema)
                else:
                    merged[key] = None
        
        return merged
    
    def _merge_single_result(self, merged_data: dict, new_data: dict, schema: dict) -> dict:
        """Merge a single extraction result into the merged data"""
        
        if not isinstance(new_data, dict):
            return merged_data
        
        for key, value in new_data.items():
            if key not in merged_data:
                merged_data[key] = value
                continue
            
            # Get the schema for this field
            field_schema = schema.get('properties', {}).get(key, {})
            field_type = field_schema.get('type')
            
            if field_type == 'array' and isinstance(value, list):
                # Merge arrays, avoiding duplicates
                if not isinstance(merged_data[key], list):
                    merged_data[key] = []
                for item in value:
                    if item not in merged_data[key]:
                        merged_data[key].append(item)
            
            elif field_type == 'object' and isinstance(value, dict):
                # Recursively merge objects
                if not isinstance(merged_data[key], dict):
                    merged_data[key] = {}
                merged_data[key] = self._merge_single_result(
                    merged_data[key], value, field_schema
                )
            
            elif merged_data[key] is None and value is not None:
                # Replace null with actual value
                merged_data[key] = value
        
        return merged_data
    
    def _post_process_merged_data(self, merged_data: dict, schema: dict) -> dict:
        """Post-process merged data for consistency"""
        
        # Remove None values for optional fields
        if 'required' in schema:
            required_fields = set(schema['required'])
            for key in list(merged_data.keys()):
                if key not in required_fields and merged_data[key] is None:
                    del merged_data[key]
        
        # Sort arrays if they contain sortable items
        for key, value in merged_data.items():
            if isinstance(value, list) and value:
                try:
                    if all(isinstance(item, (str, int, float)) for item in value):
                        merged_data[key] = sorted(list(set(value)))
                except:
                    pass  # Keep original order if sorting fails
        
        return merged_data
    
    def _calculate_total_usage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total token usage and cost"""
        total_usage = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'successful_calls': 0,
            'failed_calls': 0
        }
        
        for result in results:
            if result['success'] and 'usage' in result:
                usage = result['usage']
                total_usage['total_input_tokens'] += usage.get('input_tokens', 0)
                total_usage['total_output_tokens'] += usage.get('output_tokens', 0)
                total_usage['total_tokens'] += usage.get('total_tokens', 0)
                total_usage['total_cost'] += usage.get('estimated_cost', 0.0)
                total_usage['successful_calls'] += 1
            else:
                total_usage['failed_calls'] += 1
        
        return total_usage
    
    def iterative_extraction(self, 
                           schema: dict, 
                           text: str, 
                           model: str,
                           max_iterations: int = 3) -> Dict[str, Any]:
        """Perform iterative extraction with progressive refinement"""
        
        self.log_step("Iterative Extraction", "info", f"Starting {max_iterations} iteration(s)")
        
        # First pass: High-level extraction
        high_level_schema = self._create_high_level_schema(schema)
        
        result = self.model_interface.call_model(
            prompt=self.create_extraction_prompt(high_level_schema, text[:5000]),  # Limit first pass
            model=model,
            json_mode=True
        )
        
        if not result['success']:
            self.log_step("Iterative Extraction", "error", "Failed on first pass")
            return {'success': False, 'error': result['error']}
        
        is_valid, parsed_data, _ = self.model_interface.validate_json_output(result['content'])
        if not is_valid:
            self.log_step("Iterative Extraction", "error", "Invalid JSON on first pass")
            return {'success': False, 'error': 'Invalid JSON structure'}
        
        self.log_step("First Pass", "success", "High-level extraction completed")
        
        # Subsequent passes: Detail refinement
        current_data = parsed_data
        
        for iteration in range(2, max_iterations + 1):
            refinement_prompt = self._create_refinement_prompt(
                schema, current_data, text, iteration
            )
            
            result = self.model_interface.call_model(
                prompt=refinement_prompt,
                model=model,
                json_mode=True
            )
            
            if result['success']:
                is_valid, refined_data, _ = self.model_interface.validate_json_output(result['content'])
                if is_valid:
                    current_data = self._merge_refinement(current_data, refined_data)
                    self.log_step(f"Pass {iteration}", "success", "Refinement completed")
                else:
                    self.log_step(f"Pass {iteration}", "warning", "Invalid JSON, keeping previous result")
            else:
                self.log_step(f"Pass {iteration}", "warning", "API call failed, keeping previous result")
        
        return {
            'success': True,
            'data': current_data,
            'iterations_completed': max_iterations
        }
    
    def _create_high_level_schema(self, full_schema: dict) -> dict:
        """Create a simplified schema for first pass extraction"""
        if 'properties' not in full_schema:
            return full_schema
        
        high_level_props = {}
        for key, prop in full_schema['properties'].items():
            if prop.get('type') in ['string', 'number', 'integer', 'boolean']:
                high_level_props[key] = prop
            elif prop.get('type') == 'array':
                # Simplify arrays to just indicate they exist
                high_level_props[key] = {'type': 'array', 'items': {'type': 'string'}}
        
        return {
            'type': 'object',
            'properties': high_level_props
        }
    
    def _create_refinement_prompt(self, schema: dict, current_data: dict, text: str, iteration: int) -> str:
        """Create prompt for refinement iterations"""
        
        current_json = json.dumps(current_data, indent=2)
        schema_str = json.dumps(schema, indent=2)
        
        return f"""
REFINEMENT PASS {iteration}

You are refining previously extracted data to be more complete and accurate.

TARGET SCHEMA:
{schema_str}

CURRENT EXTRACTED DATA:
{current_json}

ORIGINAL TEXT:
{text}

INSTRUCTIONS:
1. Review the current extracted data against the target schema
2. Look for missing fields, incomplete arrays, or inaccurate values
3. Add any missing information found in the original text
4. Correct any inaccuracies you identify
5. Ensure all required fields are populated
6. Return the improved, complete JSON structure

Return ONLY the refined JSON, no additional text.
"""
    
    def _merge_refinement(self, current_data: dict, refined_data: dict) -> dict:
        """Merge refinement data with current data"""
        
        def deep_merge(base: dict, update: dict) -> dict:
            result = base.copy()
            for key, value in update.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    elif isinstance(result[key], list) and isinstance(value, list):
                        # Merge lists, avoiding duplicates
                        combined = result[key] + [item for item in value if item not in result[key]]
                        result[key] = combined
                    elif value is not None:  # Update with non-null values
                        result[key] = value
                else:
                    result[key] = value
            return result
        
        return deep_merge(current_data, refined_data)
    
    def get_processing_logs(self) -> List[Dict[str, Any]]:
        """Get all processing logs"""
        return self.extraction_logs.copy()
    
    def clear_logs(self):
        """Clear processing logs"""
        self.extraction_logs.clear()
