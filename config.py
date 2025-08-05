import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model Configuration
    MODELS = {
        'gpt-4.1': {
            'name': 'GPT-4.1',
            'max_tokens': 128000,
            'cost_per_1k_input': 0.03,
            'cost_per_1k_output': 0.06,
            'description': 'Most advanced model with superior reasoning and coding capabilities',
            'best_for': 'Complex JSON schemas, nested structures, programming tasks',
            'pros': ['Highest accuracy', 'Best reasoning', 'Handles complex tasks'],
            'cons': ['Most expensive', 'Slower response times']
        },
        'gpt-4o-mini': {
            'name': 'GPT-4o Mini',
            'max_tokens': 128000,
            'cost_per_1k_input': 0.00015,
            'cost_per_1k_output': 0.0006,
            'description': 'Fast and affordable model for focused tasks',
            'best_for': 'Simple to moderate JSON extraction, high-volume processing',
            'pros': ['Very cost-effective', 'Fast response', 'Good for simple tasks'],
            'cons': ['Limited complex reasoning', 'May miss nuanced patterns']
        },
        'o3': {
            'name': 'GPT-O3',
            'max_tokens': 128000,
            'cost_per_1k_input': 0.02,
            'cost_per_1k_output': 0.04,
            'description': 'Balanced model with good performance and moderate cost',
            'best_for': 'General-purpose JSON extraction, balanced cost/performance',
            'pros': ['Good balance of cost and capability', 'Reliable performance'],
            'cons': ['Not as advanced as GPT-4.1', 'More expensive than mini']
        }
    }
    
    # Processing Configuration
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    MAX_RETRIES = 3
    TEMPERATURE = 0.1
    
    # File Configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_TEXT_EXTENSIONS = ['.txt', '.md', '.csv']
    LOGS_DIR = 'logs'
    TEMP_DIR = 'temp'

