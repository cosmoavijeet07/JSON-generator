from typing import Dict, Any, Optional
import json

class SimplePipeline:
    def __init__(self, logger, session_manager, llm_interface, json_extractor, 
                 schema_validator, prompt_engine, token_estimator):
        self.logger = logger
        self.session_manager = session_manager
        self.llm = llm_interface
        self.extractor = json_extractor
        self.validator = schema_validator
        self.prompt_engine = prompt_engine
        self.token_estimator = token_estimator
        self.max_retries = 3
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run simple extraction pipeline"""
        
        schema = input_data["schema"]
        text = input_data["text"]
        model = input_data.get("model", "gpt-4-turbo")
        session_id = input_data.get("session_id")
        
        self.logger.log("pipeline", "start", {"type": "simple", "model": model})
        
        # Validate schema
        valid, error = self.validator.validate_schema(schema)
        if not valid:
            self.logger.log("error", "schema_validation", {"error": error}, level="ERROR")
            return {"success": False, "error": f"Schema validation failed: {error}"}
        
        # Estimate tokens
        token_count = self.token_estimator.estimate_tokens(
            json.dumps(schema) + text, model
        )
        self.logger.log("metrics", "token_estimation", {"tokens": token_count})
        
        # Try extraction with retries
        result = None
        last_error = None
        
        for attempt in range(self.max_retries):
            self.logger.log("extraction", f"attempt_{attempt+1}", {"retry": attempt > 0})
            
            # Create prompt
            prompt = self.prompt_engine.create_prompt(
                "extraction",
                schema=json.dumps(schema, indent=2),
                text=text,
                error=last_error
            )
            
            self.logger.log("prompt", f"attempt_{attempt+1}", prompt)
            
            # Call LLM
            try:
                llm_output = self.llm.call_llm(prompt, model=model)
                self.logger.log("output", f"attempt_{attempt+1}", llm_output)
                
                # Extract JSON
                extracted = self.extractor.extract_json(llm_output)
                
                # Validate
                valid, error = self.validator.validate_instance(extracted, schema)
                
                if valid:
                    result = extracted
                    self.logger.log("extraction", "success", {"attempt": attempt + 1})
                    break
                else:
                    last_error = error
                    self.logger.log("validation", f"failed_{attempt+1}", {"error": error})
                    
            except Exception as e:
                last_error = str(e)
                self.logger.log("error", f"extraction_{attempt+1}", {"error": str(e)}, level="ERROR")
        
        # Prepare response
        if result:
            self.logger.save_output("extracted_json", result)
            
            return {
                "success": True,
                "output": result,
                "metrics": {
                    "tokens_used": token_count,
                    "attempts": attempt + 1,
                    "model": model
                }
            }
        else:
            return {
                "success": False,
                "error": last_error,
                "metrics": {
                    "tokens_used": token_count,
                    "attempts": self.max_retries,
                    "model": model
                }
            }
