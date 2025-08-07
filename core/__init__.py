"""
Enhanced JSON Extraction System Core Modules
"""

from .json_extractor import extract_json, validate_against_schema
from .llm_interface import call_llm, validate_model_availability, get_model_info
from .logger_service import logger_service, log
from .prompt_engine import create_prompt, create_adaptive_prompt, create_cascade_prompt
from .schema_analyze import is_valid_schema
from .session_manager import session_manager_instance, create_session
from .token_estimator import estimate_tokens
from .text_analyzer import text_analyzer
from .schema_analyze import schema_analyzer
from .extraction_engine import extraction_engine
from .merger_engine import merger_engine
from .embedding_manager import embedding_manager

__version__ = "2.0.0"

__all__ = [
    # JSON Processing
    'extract_json',
    'validate_against_schema',
    'is_valid_schema',
    
    # LLM Interface
    'call_llm',
    'validate_model_availability', 
    'get_model_info',
    
    # Logging and Session
    'logger_service',
    'log',
    'session_manager_instance',
    'create_session',
    
    # Prompt Engineering
    'create_prompt',
    'create_adaptive_prompt',
    'create_cascade_prompt',
    
    # Token Management
    'estimate_tokens',
    
    # Analysis and Processing Engines
    'text_analyzer',
    'schema_analyzer',
    'extraction_engine',
    'merger_engine',
    'embedding_manager'
]