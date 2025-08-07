from datetime import datetime
import os
import json
import threading
from typing import Dict, List, Any, Optional
from enum import Enum
import traceback

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LoggerService:
    """Enhanced logging service with structured logging and buffering"""
    
    def __init__(self, base_dir: str = "logs"):
        """
        Initialize logger service
        
        Args:
            base_dir: Base directory for log files
        """
        self.base_dir = base_dir
        self.log_buffer = {}
        self.lock = threading.Lock()
        self.session_metadata = {}
        
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
    
    def log(
        self, 
        session_id: str, 
        stage: str, 
        content: Any, 
        level: str = "INFO",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a message with structured format
        
        Args:
            session_id: Session identifier
            stage: Processing stage
            content: Log content (can be string, dict, or any object)
            level: Log level
            metadata: Additional metadata
        
        Returns:
            Path to log file
        """
        with self.lock:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Create session directory
            log_dir = os.path.join(self.base_dir, session_id)
            os.makedirs(log_dir, exist_ok=True)
            
            # Convert content to string if necessary
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2, default=str)
            elif isinstance(content, Exception):
                content_str = f"{type(content).__name__}: {str(content)}\n{traceback.format_exc()}"
            else:
                content_str = str(content)
            
            # Create structured log entry
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "timestamp_unix": timestamp.timestamp(),
                "level": level,
                "stage": stage,
                "content": content_str,
                "metadata": metadata or {}
            }
            
            # Add to buffer
            if session_id not in self.log_buffer:
                self.log_buffer[session_id] = []
            self.log_buffer[session_id].append(log_entry)
            
            # Write to file
            log_file = os.path.join(log_dir, f"{stage}_{timestamp_str}.log")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, indent=2, default=str))
            
            # Also write to consolidated log
            consolidated_file = os.path.join(log_dir, "session.log")
            with open(consolidated_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
            
            return log_file
    
    def get_session_logs(
        self, 
        session_id: str, 
        stage: Optional[str] = None,
        level: Optional[str] = None
    ) -> List[Dict]:
        """
        Get logs for a session
        
        Args:
            session_id: Session identifier
            stage: Filter by stage (optional)
            level: Filter by level (optional)
        
        Returns:
            List of log entries
        """
        with self.lock:
            logs = self.log_buffer.get(session_id, [])
            
            # Apply filters
            if stage:
                logs = [log for log in logs if log['stage'] == stage]
            
            if level:
                logs = [log for log in logs if log['level'] == level]
            
            return logs
    
    def export_session_logs(
        self, 
        session_id: str, 
        format: str = "json",
        include_metadata: bool = True
    ) -> str:
        """
        Export session logs in specified format
        
        Args:
            session_id: Session identifier
            format: Export format ('json', 'text', 'csv')
            include_metadata: Include metadata in export
        
        Returns:
            Exported logs as string
        """
        logs = self.get_session_logs(session_id)
        
        if format == "json":
            return json.dumps(logs, indent=2, default=str)
        
        elif format == "text":
            text_logs = []
            for entry in logs:
                timestamp = entry['timestamp']
                level = entry['level']
                stage = entry['stage']
                content = entry['content']
                
                log_line = f"[{timestamp}] {level:8} | {stage:20} | {content}"
                
                if include_metadata and entry.get('metadata'):
                    log_line += f" | metadata: {json.dumps(entry['metadata'])}"
                
                text_logs.append(log_line)
            
            return "\n".join(text_logs)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(
                output, 
                fieldnames=['timestamp', 'level', 'stage', 'content', 'metadata']
            )
            writer.writeheader()
            
            for entry in logs:
                row = {
                    'timestamp': entry['timestamp'],
                    'level': entry['level'],
                    'stage': entry['stage'],
                    'content': entry['content'],
                    'metadata': json.dumps(entry.get('metadata', {})) if include_metadata else ''
                }
                writer.writerow(row)
            
            return output.getvalue()
        
        else:
            return str(logs)
    
    def log_performance(
        self,
        session_id: str,
        operation: str,
        duration: float,
        success: bool,
        details: Optional[Dict] = None
    ):
        """
        Log performance metrics
        
        Args:
            session_id: Session identifier
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            details: Additional details
        """
        metadata = {
            "operation": operation,
            "duration_seconds": duration,
            "duration_ms": duration * 1000,
            "success": success,
            "details": details or {}
        }
        
        self.log(
            session_id,
            "performance",
            f"Operation '{operation}' completed in {duration:.3f}s",
            "INFO" if success else "WARNING",
            metadata
        )
    
    def log_error(
        self,
        session_id: str,
        stage: str,
        error: Exception,
        context: Optional[Dict] = None
    ):
        """
        Log an error with full traceback
        
        Args:
            session_id: Session identifier
            stage: Processing stage where error occurred
            error: Exception object
            context: Additional context
        """
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.log(
            session_id,
            stage,
            error,
            "ERROR",
            error_details
        )
    
    def log_token_usage(
        self,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost: Optional[float] = None
    ):
        """
        Log token usage for LLM calls
        
        Args:
            session_id: Session identifier
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            cost: Estimated cost (optional)
        """
        metadata = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost
        }
        
        self.log(
            session_id,
            "token_usage",
            f"Model {model}: {total_tokens} tokens used",
            "INFO",
            metadata
        )
    
    def set_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ):
        """
        Set metadata for a session
        
        Args:
            session_id: Session identifier
            metadata: Session metadata
        """
        with self.lock:
            self.session_metadata[session_id] = metadata
            
            # Save to file
            session_dir = os.path.join(self.base_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            metadata_file = os.path.join(session_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get metadata for a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session metadata
        """
        with self.lock:
            if session_id in self.session_metadata:
                return self.session_metadata[session_id]
            
            # Try to load from file
            metadata_file = os.path.join(self.base_dir, session_id, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    self.session_metadata[session_id] = metadata
                    return metadata
            
            return {}
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of session logs
        
        Args:
            session_id: Session identifier
        
        Returns:
            Summary statistics
        """
        logs = self.get_session_logs(session_id)
        
        if not logs:
            return {"total_logs": 0}
        
        # Count by level
        level_counts = {}
        for log in logs:
            level = log['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by stage
        stage_counts = {}
        for log in logs:
            stage = log['stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Time range
        timestamps = [log['timestamp_unix'] for log in logs]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        return {
            "total_logs": len(logs),
            "level_counts": level_counts,
            "stage_counts": stage_counts,
            "duration_seconds": duration,
            "start_time": datetime.fromtimestamp(min(timestamps)).isoformat() if timestamps else None,
            "end_time": datetime.fromtimestamp(max(timestamps)).isoformat() if timestamps else None,
            "has_errors": level_counts.get("ERROR", 0) > 0,
            "has_warnings": level_counts.get("WARNING", 0) > 0
        }
    
    def clear_session_logs(self, session_id: str):
        """
        Clear logs for a session from memory
        
        Args:
            session_id: Session identifier
        """
        with self.lock:
            if session_id in self.log_buffer:
                del self.log_buffer[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
    
    def cleanup_old_logs(self, days: int = 7):
        """
        Clean up logs older than specified days
        
        Args:
            days: Number of days to keep logs
        """
        import shutil
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for session_dir in os.listdir(self.base_dir):
            session_path = os.path.join(self.base_dir, session_dir)
            
            if os.path.isdir(session_path):
                # Check modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
                
                if mtime < cutoff_date:
                    try:
                        shutil.rmtree(session_path)
                        print(f"Cleaned up old logs: {session_dir}")
                    except Exception as e:
                        print(f"Error cleaning up {session_dir}: {e}")

# Initialize global logger instance
logger_service = LoggerService()

# Convenience function for simple logging
def log(session_id: str, stage: str, content: Any, level: str = "INFO", metadata: Optional[Dict] = None):
    """
    Convenience function for logging
    
    Args:
        session_id: Session identifier
        stage: Processing stage
        content: Log content
        level: Log level
        metadata: Additional metadata
    """
    return logger_service.log(session_id, stage, content, level, metadata)