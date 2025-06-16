
#!/usr/bin/env python3
"""
Performance monitoring and optimization for PII detection system
"""

import time
import functools
from typing import Dict, Any, Callable
from collections import defaultdict, deque
import threading
import psutil
import os

class PerformanceMonitor:
    """Thread-safe performance monitoring system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        with self.lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
    
    def increment_counter(self, name: str, count: int = 1):
        """Increment a counter metric"""
        with self.lock:
            self.counters[name] += count
    
    def time_function(self, function_name: str):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_metric(f"{function_name}_execution_time", execution_time)
                    self.increment_counter(f"{function_name}_calls")
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.lock:
            stats = {
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                },
                'counters': dict(self.counters),
                'metrics': {}
            }
            
            # Calculate metric summaries
            for name, values in self.metrics.items():
                if values:
                    recent_values = [v['value'] for v in values]
                    stats['metrics'][name] = {
                        'count': len(recent_values),
                        'avg': sum(recent_values) / len(recent_values),
                        'min': min(recent_values),
                        'max': max(recent_values),
                        'latest': recent_values[-1] if recent_values else 0
                    }
            
            return stats

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# LRU Cache implementation for predictions
class LRUCache:
    """Simple LRU cache for caching predictions"""
    
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
    
    def get(self, key: str):
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

# Global prediction cache
prediction_cache = LRUCache(capacity=2000)

def cached_prediction(cache_key_prefix: str):
    """Decorator for caching predictions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = f"{cache_key_prefix}_{hash(str(args[1:]) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = prediction_cache.get(cache_key)
            if cached_result is not None:
                performance_monitor.increment_counter(f"{cache_key_prefix}_cache_hits")
                return cached_result
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            prediction_cache.put(cache_key, result)
            performance_monitor.increment_counter(f"{cache_key_prefix}_cache_misses")
            
            return result
        return wrapper
    return decorator

def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    performance_monitor.record_metric(f"memory_usage_{stage}", memory_mb)
    return memory_mb

def optimize_model_memory():
    """Optimize model memory usage"""
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    memory_after = log_memory_usage("after_cleanup")
    return memory_after
