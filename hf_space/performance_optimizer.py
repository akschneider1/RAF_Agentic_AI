
import time
import functools
from typing import Any, Callable

class PerformanceMonitor:
    """Simplified performance monitoring for HF Space"""
    
    def __init__(self):
        self.stats = {}
    
    def time_function(self, name: str):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if name not in self.stats:
                    self.stats[name] = []
                self.stats[name].append(end_time - start_time)
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self):
        """Get performance statistics"""
        return self.stats

# Global instance
performance_monitor = PerformanceMonitor()

def cached_prediction(cache_type: str):
    """Simplified caching decorator"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simple cache key based on text content
            cache_key = str(args) + str(kwargs)
            
            if cache_key in cache:
                return cache[cache_key]
            
            result = func(*args, **kwargs)
            cache[cache_key] = result
            
            # Keep cache size manageable
            if len(cache) > 1000:
                cache.clear()
            
            return result
        return wrapper
    return decorator
