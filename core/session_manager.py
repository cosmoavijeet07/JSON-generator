import os
import uuid
import json
from datetime import datetime

def create_session():
    session_id = str(uuid.uuid4())
    os.makedirs(f"logs/{session_id}", exist_ok=True)
    
    # Create session metadata
    metadata = {
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'processing_mode': 'enhanced',
        'pipeline_stages': []
    }
    
    with open(f"logs/{session_id}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return session_id

def log_pipeline_stage(session_id: str, stage: str, data: dict):
    """Log pipeline stage information"""
    metadata_path = f"logs/{session_id}/metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'pipeline_stages': []}
    
    stage_info = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    metadata['pipeline_stages'].append(stage_info)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)