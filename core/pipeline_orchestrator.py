import json
from typing import Dict, List, Any
from .text_preprocessor import TextPreprocessor
from .schema_decomposer import SchemaDecomposer
from .chunk_processor import ChunkProcessor
from .logger_service import log

class PipelineOrchestrator:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.schema_decomposer = SchemaDecomposer()
        self.chunk_processor = ChunkProcessor()
    
    def process_complex_extraction(self, text: str, schema: Dict, session_id: str, 
                                 model: str = "gpt-4.1-2025-04-14") -> Dict:
        """Main pipeline for complex text extraction"""
        log(session_id, "pipeline_start", "Starting complex extraction pipeline")

        # Step 1: Preprocess and analyze text structure
        log(session_id, "preprocessing", "Analyzing document structure")
        structure = self.preprocessor.detect_document_structure(text)

        # Step 2: Create semantic chunks
        log(session_id, "chunking", "Creating semantic chunks")
        chunks = self.preprocessor.semantic_chunking(text, max_chunk_size=2000)

        # Step 3: Classify content types
        for chunk in chunks:
            chunk['content_type'] = self.preprocessor.classify_content_type(chunk['text'])

        log(session_id, "chunks_created", json.dumps({
            'total_chunks': len(chunks),
            'content_types': [c['content_type'] for c in chunks]
        }))

        # Step 4: Decompose schema if complex
        sub_schemas = self.schema_decomposer.decompose_schema(schema)
        log(session_id, "schema_decomposition", f"Created {len(sub_schemas)} sub-schemas")

        # Step 5: Process chunks with different strategies based on schema complexity
        if len(sub_schemas) > 1:
            final_result = self._process_with_sub_schemas(chunks, sub_schemas, session_id, model)
        else:
            final_result = self.chunk_processor.process_chunks(chunks, schema, model)

        # Step 6: Skip Quality Assurance – directly return result
        return {
            'result': final_result,
            'metadata': {
                'chunks_processed': len(chunks),
                'document_structure': structure,
                'confidence_score': 1.0,  # Default full confidence
                'validation_issues': []   # No QA = no issues
            }
        }

    def _process_with_sub_schemas(self, chunks: List[Dict], sub_schemas: List[Dict], 
                                  session_id: str, model: str) -> Dict:
        results = {}
        for sub_schema in sub_schemas:
            property_name = sub_schema.get('_property_name')
            if property_name:
                relevant_chunks = self._filter_relevant_chunks(chunks, property_name)
                if relevant_chunks:
                    chunk_processor = ChunkProcessor()
                    sub_result = chunk_processor.process_chunks(relevant_chunks, sub_schema, model)
                    if sub_result and property_name in sub_result:
                        results[property_name] = sub_result[property_name]
        return results

    def _filter_relevant_chunks(self, chunks: List[Dict], property_name: str) -> List[Dict]:
        relevant = []
        property_keywords = property_name.lower().split('_')
        for chunk in chunks:
            chunk_text_lower = chunk['text'].lower()
            if any(keyword in chunk_text_lower for keyword in property_keywords):
                relevant.append(chunk)
        return relevant if relevant else chunks

    def _fallback_extraction(self, text: str, schema: Dict, model: str) -> Dict:
        chunks = []
        chunk_size = 1500
        overlap = 200
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({'text': chunk_text, 'content_type': 'general'})
        processor = ChunkProcessor()
        return processor.process_chunks(chunks, schema, model)

    
    