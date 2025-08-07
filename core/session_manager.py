import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import threading

class SessionStatus(Enum):
    """Session status enumeration"""
    CREATED = "created"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SessionManager:
    """Enhanced session management with metadata and state tracking"""
    
    def __init__(self, base_dir: str = "logs"):
        """
        Initialize session manager
        
        Args:
            base_dir: Base directory for session data
        """
        self.base_dir = base_dir
        self.sessions = {}
        self.lock = threading.Lock()
        
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Load existing sessions
        self._load_existing_sessions()
    
    def create_session(
        self,
        pipeline_type: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new session with metadata
        
        Args:
            pipeline_type: Type of pipeline (simple/extensive)
            model: Model being used
            metadata: Additional metadata
        
        Returns:
            Session ID
        """
        with self.lock:
            session_id = str(uuid.uuid4())
            session_dir = os.path.join(self.base_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Create session metadata
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": SessionStatus.CREATED.value,
                "pipeline_type": pipeline_type,
                "model": model,
                "metadata": metadata or {},
                "progress": {
                    "current_step": "",
                    "total_steps": 0,
                    "completed_steps": 0,
                    "percentage": 0
                },
                "statistics": {
                    "start_time": None,
                    "end_time": None,
                    "duration_seconds": None,
                    "tokens_used": 0,
                    "chunks_processed": 0,
                    "partitions_processed": 0
                },
                "results": {
                    "output_file": None,
                    "log_file": None,
                    "success": None,
                    "errors": []
                }
            }
            
            # Save metadata
            self._save_session_data(session_id, session_data)
            
            # Store in memory
            self.sessions[session_id] = session_data
            
            return session_id
    
    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ):
        """
        Update session data
        
        Args:
            session_id: Session identifier
            updates: Updates to apply
            merge: Whether to merge with existing data
        """
        with self.lock:
            if session_id not in self.sessions:
                # Try to load from disk
                session_data = self._load_session_data(session_id)
                if session_data:
                    self.sessions[session_id] = session_data
                else:
                    raise ValueError(f"Session {session_id} not found")
            
            if merge:
                # Merge updates
                self._deep_merge(self.sessions[session_id], updates)
            else:
                # Replace
                self.sessions[session_id].update(updates)
            
            # Update timestamp
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
            
            # Save to disk
            self._save_session_data(session_id, self.sessions[session_id])
    
    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        message: Optional[str] = None
    ):
        """
        Update session status
        
        Args:
            session_id: Session identifier
            status: New status
            message: Optional status message
        """
        updates = {"status": status.value}
        
        if message:
            updates["status_message"] = message
        
        # Update timestamps based on status
        if status == SessionStatus.PROCESSING:
            updates["statistics"] = {"start_time": datetime.now().isoformat()}
        elif status in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
            stats = self.get_session(session_id).get("statistics", {})
            if stats.get("start_time"):
                end_time = datetime.now()
                start_time = datetime.fromisoformat(stats["start_time"])
                duration = (end_time - start_time).total_seconds()
                
                updates["statistics"] = {
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration
                }
        
        self.update_session(session_id, updates)
    
    def update_progress(
        self,
        session_id: str,
        current_step: str,
        completed_steps: int,
        total_steps: int
    ):
        """
        Update session progress
        
        Args:
            session_id: Session identifier
            current_step: Current step description
            completed_steps: Number of completed steps
            total_steps: Total number of steps
        """
        percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        updates = {
            "progress": {
                "current_step": current_step,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "percentage": round(percentage, 2)
            }
        }
        
        self.update_session(session_id, updates)
    
    def add_error(
        self,
        session_id: str,
        error: str,
        stage: Optional[str] = None
    ):
        """
        Add an error to session
        
        Args:
            session_id: Session identifier
            error: Error message
            stage: Stage where error occurred
        """
        session = self.get_session(session_id)
        errors = session.get("results", {}).get("errors", [])
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "stage": stage
        }
        
        errors.append(error_entry)
        
        self.update_session(session_id, {"results": {"errors": errors}})
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get session data
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data dictionary
        """
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].copy()
            
            # Try to load from disk
            session_data = self._load_session_data(session_id)
            if session_data:
                self.sessions[session_id] = session_data
                return session_data.copy()
            
            return {}
    
    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List sessions
        
        Args:
            status: Filter by status
            limit: Maximum number of sessions to return
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        # List from disk
        for session_dir in os.listdir(self.base_dir):
            session_path = os.path.join(self.base_dir, session_dir, "metadata.json")
            
            if os.path.exists(session_path):
                try:
                    with open(session_path, "r") as f:
                        session_data = json.load(f)
                        
                        # Apply filter
                        if status and session_data.get("status") != status.value:
                            continue
                        
                        # Create summary
                        summary = {
                            "session_id": session_data["session_id"],
                            "created_at": session_data["created_at"],
                            "status": session_data["status"],
                            "pipeline_type": session_data.get("pipeline_type"),
                            "model": session_data.get("model"),
                            "progress": session_data.get("progress", {}).get("percentage", 0)
                        }
                        
                        sessions.append(summary)
                        
                        if len(sessions) >= limit:
                            break
                
                except Exception as e:
                    print(f"Error loading session {session_dir}: {e}")
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return sessions
    
    def delete_session(self, session_id: str):
        """
        Delete a session
        
        Args:
            session_id: Session identifier
        """
        import shutil
        
        with self.lock:
            # Remove from memory
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Remove from disk
            session_dir = os.path.join(self.base_dir, session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
    
    def _save_session_data(self, session_id: str, data: Dict[str, Any]):
        """Save session data to disk"""
        session_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        metadata_file = os.path.join(session_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from disk"""
        metadata_file = os.path.join(self.base_dir, session_id, "metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
        
        return None
    
    def _load_existing_sessions(self):
        """Load existing sessions from disk"""
        try:
            for session_dir in os.listdir(self.base_dir):
                session_path = os.path.join(self.base_dir, session_dir)
                if os.path.isdir(session_path):
                    session_data = self._load_session_data(session_dir)
                    if session_data:
                        self.sessions[session_dir] = session_data
        except Exception as e:
            print(f"Error loading existing sessions: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

# Initialize global instance
session_manager_instance = SessionManager()

# Convenience function
def create_session(pipeline_type: Optional[str] = None, model: Optional[str] = None) -> str:
    """Create a new session"""
    return session_manager_instance.create_session(pipeline_type, model)