from typing import Dict, Any, Optional, Callable
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PipelineType(Enum):
    SIMPLE = "simple"
    EXTENSIVE = "extensive"

class PipelineManager:
    def __init__(self, logger_service, session_manager):
        self.logger = logger_service
        self.session_manager = session_manager
        self.pipelines = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def register_pipeline(self, name: str, pipeline_class):
        """Register a pipeline class"""
        self.pipelines[name] = pipeline_class
    
    def get_pipeline(self, pipeline_type: PipelineType, **kwargs):
        """Get pipeline instance"""
        pipeline_class = self.pipelines.get(pipeline_type.value)
        if not pipeline_class:
            raise ValueError(f"Pipeline {pipeline_type.value} not found")
        
        return pipeline_class(
            logger=self.logger,
            session_manager=self.session_manager,
            **kwargs
        )
    
    async def run_pipeline_async(self, pipeline, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, pipeline.run, input_data)
    
    def run_pipeline(self, pipeline_type: PipelineType, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run pipeline synchronously"""
        pipeline = self.get_pipeline(pipeline_type, **kwargs)
        return pipeline.run(input_data)
