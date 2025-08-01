import os
import uuid

def create_session():
    session_id = str(uuid.uuid4())
    os.makedirs(f"logs/{session_id}", exist_ok=True)
    return session_id
