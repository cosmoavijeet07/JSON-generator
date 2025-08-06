from typing import Dict, Any, List, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ExtensivePipeline:
    def __init__(self, logger, session_manager, llm_interface, json_extractor,
                 schema_validator, prompt_engine, token_estimator,
                 text_processor, schema_processor, chunk_manager,
                 merge_manager, embedding_service):
        self.logger = logger
        self.session_manager = session_manager
        self.llm = llm_interface
        self.extractor = json_extractor
        self.validator = schema_validator
        self.prompt_engine = prompt_engine
        self.token_estimator = token_estimator
        self.text_processor = text_processor
        self.schema_processor = schema_processor
        self.chunk_manager = chunk_manager
        self.merge_manager = merge_manager
        self.embedding_service = embedding_service
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run extensive extraction pipeline"""
        
        schema = input_data["schema"]
        text = input_data["text"]
        model = input_data.get("model", "claude-3.5-sonnet")
        session_id = input_data.get("session_id")
        num_passes = input_data.get("num_passes", 3)
        
        self.logger.log("pipeline", "start", {"type": "extensive", "model": model})
        
        try:
            # Phase 1: Analyze and preprocess
            text_analysis = self._analyze_text(text)
            schema_analysis = self._analyze_schema(schema)
            
            # Phase 2: Chunking and partitioning
            text_chunks = self._chunk_text(text, text_analysis)
            schema_partitions = self._partition_schema(schema, schema_analysis)
            
            # Phase 3: Create embeddings
            text_embedding = self._create_embeddings(text)
            schema_embedding = self._create_embeddings(json.dumps(schema))
            
            # Phase 4: Multi-pass extraction
            extraction_results = self._multi_pass_extraction(
                text_chunks, schema_partitions, model, num_passes,
                text_embedding, schema_embedding
            )
            
            # Phase 5: Merge results
            merged_result = self._merge_results(extraction_results, schema)
            
            # Phase 6: Validate and refine
            final_result = self._validate_and_refine(merged_result, schema, text, model)
            
            self.logger.save_output("final_extracted_json", final_result)
            
            return {
                "success": True,
                "output": final_result,
                "metrics": {
                    "text_chunks": len(text_chunks),
                    "schema_partitions": len(schema_partitions),
                    "extraction_passes": num_passes,
                    "model": model
                }
            }
            
        except Exception as e:
            self.logger.log("error", "pipeline_failure", {"error": str(e)}, level="ERROR")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text document"""
        self.logger.log("analysis", "text_start", {"length": len(text)})
        
        analysis = self.text_processor.analyze_document(text)
        
        # Get LLM analysis
        prompt = self.prompt_engine.create_prompt(
            "analysis",
            text=text[:10000]  # Limit for analysis
        )
        
        llm_analysis = self.llm.call_llm(prompt)
        
        try:
            llm_analysis_json = json.loads(llm_analysis)
            analysis.update(llm_analysis_json)
        except:
            pass
        
        self.logger.log("analysis", "text_complete", analysis)
        return analysis
    
    def _analyze_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema complexity"""
        self.logger.log("analysis", "schema_start", {})
        
        analysis = self.schema_processor.analyze_schema(schema)
        
        self.logger.log("analysis", "schema_complete", analysis)
        return analysis
    
    def _chunk_text(self, text: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text based on analysis"""
        self.logger.log("chunking", "start", {})
        
        # Generate chunking code using LLM
        prompt = self.prompt_engine.create_prompt(
            "chunking",
            context={"analysis": analysis}
        )
        
        chunking_code = self.llm.call_llm(prompt)
        
        # For safety, use default chunking
        # In production, you might execute the generated code in a sandbox
        chunk_size = analysis.get("recommended_chunk_size", 2000)
        chunks = self.chunk_manager.chunk_text(text, method="semantic", chunk_size=chunk_size)
        
        self.logger.log("chunking", "complete", {"num_chunks": len(chunks)})
        return chunks
    
    def _partition_schema(self, schema: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Partition schema if needed"""
        self.logger.log("partitioning", "start", {})
        
        if analysis.get("needs_partitioning"):
            partitions = self.schema_processor.partition_schema(
                schema, 
                strategy=analysis.get("partition_strategy", "auto")
            )
        else:
            partitions = [schema]
        
        self.logger.log("partitioning", "complete", {"num_partitions": len(partitions)})
        return partitions
    
    def _create_embeddings(self, text: str) -> Dict[str, Any]:
        """Create embeddings for text"""
        self.logger.log("embedding", "start", {"length": len(text)})
        
        embedding_data = self.embedding_service.create_document_embedding(text)
        
        self.logger.log("embedding", "complete", {"num_chunks": embedding_data["num_chunks"]})
        return embedding_data
    
    def _multi_pass_extraction(self, text_chunks: List[Dict[str, Any]], 
                              schema_partitions: List[Dict[str, Any]],
                              model: str, num_passes: int,
                              text_embedding: Dict[str, Any],
                              schema_embedding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform multi-pass extraction"""
        self.logger.log("extraction", "multi_pass_start", {"passes": num_passes})
        
        all_results = []
        
        for pass_num in range(num_passes):
            self.logger.log("extraction", f"pass_{pass_num+1}_start", {})
            
            pass_results = []
            
            for chunk in text_chunks:
                for partition in schema_partitions:
                    result = self._extract_single(
                        chunk["content"],
                        partition,
                        model,
                        context={
                            "pass": pass_num + 1,
                            "chunk_id": chunk["chunk_id"],
                            "previous_results": pass_results[-1] if pass_results else None
                        }
                    )
                    
                    if result:
                        pass_results.append(result)
            
            all_results.extend(pass_results)
            
            self.logger.log("extraction", f"pass_{pass_num+1}_complete", 
                          {"results": len(pass_results)})
        
        return all_results
    
    def _extract_single(self, text: str, schema: Dict[str, Any], 
                       model: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract from single chunk with single schema partition"""
        
        prompt = self.prompt_engine.create_prompt(
            "extraction",
            schema=json.dumps(schema, indent=2),
            text=text,
            context=context
        )
        
        try:
            output = self.llm.call_llm(prompt, model=model)
            extracted = self.extractor.extract_json(output)
            
            # Quick validation
            if isinstance(extracted, dict):
                return extracted
        except Exception as e:
            self.logger.log("error", "single_extraction", {"error": str(e)})
        
        return None
    
    def _merge_results(self, results: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all extraction results"""
        self.logger.log("merging", "start", {"num_results": len(results)})
        
        if not results:
            return {}
        
        # Use intelligent merging
        merged = self.merge_manager.merge_outputs(results, schema, strategy="hierarchical")
        
        self.logger.log("merging", "complete", {})
        return merged
    
    def _validate_and_refine(self, result: Dict[str, Any], schema: Dict[str, Any],
                            original_text: str, model: str) -> Dict[str, Any]:
        """Validate and refine the final result"""
        self.logger.log("refinement", "start", {})
        
        # Validate
        valid, error = self.validator.validate_instance(result, schema)
        
        if not valid:
            # Try to fix validation errors
            prompt = self.prompt_engine.create_prompt(
                "validation",
                schema=json.dumps(schema, indent=2),
                text=original_text[:5000],
                error=error,
                context={"extracted_json": result}
            )
            
            try:
                refined_output = self.llm.call_llm(prompt, model=model)
                refined_json = self.extractor.extract_json(refined_output)
                
                # Validate again
                valid, error = self.validator.validate_instance(refined_json, schema)
                if valid:
                    result = refined_json
                    self.logger.log("refinement", "success", {})
            except Exception as e:
                self.logger.log("error", "refinement", {"error": str(e)})
        
        self.logger.log("refinement", "complete", {"valid": valid})
        return result
