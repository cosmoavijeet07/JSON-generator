from datetime import datetime
import os

def log(session_id, stage, content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{session_id}"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/{stage}_{timestamp}.log", "w", encoding="utf-8") as f:
        f.write(content)
