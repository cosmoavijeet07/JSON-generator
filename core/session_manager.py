import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

class SessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session_data = {
            "session_id": session_id,
            "created_at": timestamp,
            "status": "active",
            "metadata": metadata or {},
            "pipeline": None,
            "results": None
        }
        
        self.active_sessions[session_id] = session_data
        
        # Create session directory
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save session metadata
        with open(session_dir / "session.json", "w") as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session_file = self.base_dir / session_id / "session.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                return json.load(f)
        
        return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session data"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = self.get_session(session_id)
        
        if self.active_sessions[session_id]:
            self.active_sessions[session_id].update(updates)
            
            # Save to disk
            session_dir = self.base_dir / session_id
            with open(session_dir / "session.json", "w") as f:
                json.dump(self.active_sessions[session_id], f, indent=2)
    
    def end_session(self, session_id: str, results: Optional[Dict[str, Any]] = None):
        """End a session"""
        self.update_session(session_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": results
        })
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        sessions = []
        
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    with open(session_file, "r") as f:
                        sessions.append(json.load(f))
        
        return sessions