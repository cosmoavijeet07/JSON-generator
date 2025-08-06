import os
import json
import datetime
from typing import Any, Optional
from pathlib import Path

class LoggerService:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.log_buffer = []
    
    def start_session(self, session_id: str):
        """Start a new logging session"""
        self.current_session = session_id
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(exist_ok=True)
        self.log_file = self.session_dir / "session.log"
        self.log_buffer = []
        
        self.log("session", "start", {"session_id": session_id})
    
    def log(self, category: str, event: str, data: Any = None, level: str = "INFO"):
        """Log an event"""
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "category": category,
            "event": event,
            "data": data
        }
        
        self.log_buffer.append(log_entry)
        
        # Write to file
        if self.current_session:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        # Also write category-specific logs
        if category in ["prompt", "output", "error"]:
            self._write_category_log(category, event, data)
    
    def _write_category_log(self, category: str, event: str, data: Any):
        """Write category-specific log files"""
        if not self.current_session:
            return
        
        category_file = self.session_dir / f"{category}_{event}.txt"
        
        with open(category_file, "w", encoding="utf-8") as f:
            if isinstance(data, dict):
                f.write(json.dumps(data, indent=2))
            else:
                f.write(str(data))
    
    def get_session_logs(self) -> List[Dict[str, Any]]:
        """Get all logs for current session"""
        return self.log_buffer
    
    def save_output(self, output_type: str, data: Any, filename: Optional[str] = None):
        """Save output files"""
        if not self.current_session:
            return
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_type}_{timestamp}.json"
        
        output_file = self.session_dir / filename
        
        with open(output_file, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2)
            else:
                f.write(str(data))
        
        self.log("output", output_type, {"filename": str(output_file)})
        
        return output_file
    
    def end_session(self):
        """End the current logging session"""
        if self.current_session:
            self.log("session", "end", {
                "total_events": len(self.log_buffer),
                "session_id": self.current_session
            })
            
            # Save session summary
            summary_file = self.session_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump({
                    "session_id": self.current_session,
                    "total_events": len(self.log_buffer),
                    "start_time": self.log_buffer[0]["timestamp"] if self.log_buffer else None,
                    "end_time": self.log_buffer[-1]["timestamp"] if self.log_buffer else None,
                    "categories": list(set(log["category"] for log in self.log_buffer))
                }, f, indent=2)
        
        self.current_session = None
        self.log_buffer = []