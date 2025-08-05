import json
from typing import List, Dict, Any, Optional
from .llm_interface import call_llm
from .prompt_engine import create_focused_prompt
from .json_extractor import extract_json, validate_against_schema

class ChunkProcessor:
    def __init__(self):
        self.entity_registry = {}
        self.context_window = []
        self.max_context_size = 1000
    
    def process_chunks(self, chunks: List[Dict], schema: Dict, model: str = "gpt-4.1-2025-04-14") -> Dict:
        """Process chunks with context awareness"""
        results = []
        
        for i, chunk in enumerate(chunks):
            chunk_result = self._process_single_chunk(
                chunk, schema, model, chunk_index=i, total_chunks=len(chunks)
            )
            
            if chunk_result:
                results.append(chunk_result)
                self._update_context(chunk_result)
                self._update_entity_registry(chunk_result)
        
        return self._merge_results(results, schema)
    
    def _process_single_chunk(self, chunk: Dict, schema: Dict, model: str, 
                            chunk_index: int, total_chunks: int) -> Optional[Dict]:
        """Process a single chunk with retry logic"""
        context_info = {
            'previous_entities': self.entity_registry,
            'context_window': self.context_window[-3:],  # Last 3 chunks
            'chunk_info': {
                'index': chunk_index,
                'total': total_chunks,
                'content_type': chunk.get('content_type', 'general')
            }
        }
        
        for attempt in range(3):
            try:
                prompt = create_focused_prompt(
                    schema=json.dumps(schema),
                    text=chunk['text'],
                    context=context_info,
                    attempt=attempt
                )
                
                output = call_llm(prompt, model)
                result = extract_json(output)
                
                valid, error = validate_against_schema(schema, result)
                if valid:
                    return result
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Failed to process chunk {chunk_index}: {str(e)}")
        
        return None
    
    def _update_context(self, result: Dict):
        """Update sliding context window"""
        self.context_window.append(result)
        if len(self.context_window) > 5:
            self.context_window.pop(0)
    
    def _update_entity_registry(self, result: Dict):
        """Update entity registry for cross-reference"""
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 2:
                if key not in self.entity_registry:
                    self.entity_registry[key] = set()
                self.entity_registry[key].add(value)
    
    def _merge_results(self, results: List[Dict], schema: Dict) -> Dict:
        """Merge results from multiple chunks"""
        merged = {}
        
        for field in schema.get('properties', {}):
            field_values = []
            for result in results:
                if field in result and result[field] is not None:
                    field_values.append(result[field])
            
            if field_values:
                merged[field] = self._merge_field_values(field_values, schema['properties'][field])
        
        return merged
    
    def _merge_field_values(self, values: List[Any], field_schema: Dict) -> Any:
        """Merge values for a specific field"""
        if not values:
            return None
        
        field_type = field_schema.get('type')
        
        if field_type == 'array':
            # Flatten and deduplicate arrays
            merged_array = []
            for value in values:
                if isinstance(value, list):
                    merged_array.extend(value)
                else:
                    merged_array.append(value)
            return list(set(merged_array)) if merged_array else []
        
        elif field_type == 'object':
            # Merge objects
            merged_obj = {}
            for value in values:
                if isinstance(value, dict):
                    merged_obj.update(value)
            return merged_obj if merged_obj else None
        
        else:
            # For simple types, take the most complete/longest value
            return max(values, key=lambda x: len(str(x)) if x else 0)