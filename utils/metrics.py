import time
from typing import Dict, Any, List
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return duration"""
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.metrics[f"{name}_duration"].append(duration)
            del self.timers[name]
            return duration
        return 0
    
    def record(self, name: str, value: Any):
        """Record a metric"""
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        
        for key, values in self.metrics.items():
            if all(isinstance(v, (int, float)) for v in values):
                summary[key] = {
                    "count": len(values),
                    "total": sum(values),
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
            else:
                summary[key] = values
        
        return summary
