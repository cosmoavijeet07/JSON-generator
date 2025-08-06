import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import base64

class FileHandler:
    @staticmethod
    def read_json(filepath: str) -> Dict[str, Any]:
        """Read JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(filepath: str, data: Dict[str, Any]):
        """Write JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def read_text(filepath: str) -> str:
        """Read text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def write_text(filepath: str, text: str):
        """Write text file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    @staticmethod
    def create_directory(path: str):
        """Create directory if it doesn't exist"""
        Path(path).mkdir(parents=True, exist_ok=True)

class DisplayManager:
    @staticmethod
    def format_json(data: Dict[str, Any]) -> str:
        """Format JSON for display"""
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics for display"""
        lines = ["=== METRICS ==="]
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"\n{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def create_download_link(data: Any, filename: str) -> str:
        """Create base64 download link"""
        if isinstance(data, dict):
            data_str = json.dumps(data, indent=2)
        else:
            data_str = str(data)
        
        b64 = base64.b64encode(data_str.encode()).decode()
        return f"data:application/json;base64,{b64}"