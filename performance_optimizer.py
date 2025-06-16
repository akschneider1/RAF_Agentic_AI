
#!/usr/bin/env python3
"""
Performance optimization utilities for the PII detection system
"""

import functools
import hashlib
import time
from typing import Dict, Any, List, Optional
import torch
from collections import defaultdict, OrderedDict
import threading

class LRUCache:
    """Thread-safe LRU cache for model predictions"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.metrics[func_name].append(execution_time)
                self.counters[f"{func_name}_calls"] += 1
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for func_name, times in self.metrics.items():
            if times:
                stats[func_name] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'total_calls': len(times)
                }
        return stats

class ModelCache:
    """Intelligent model caching system"""
    
    def __init__(self):
        self.prediction_cache = LRUCache(max_size=2000)
        self.text_hash_cache = LRUCache(max_size=5000)
        self.performance_monitor = PerformanceMonitor()
    
    def get_text_hash(self, text: str) -> str:
        """Generate consistent hash for text"""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    @functools.lru_cache(maxsize=1000)
    def get_pattern_matches(self, text: str, pattern_key: str) -> str:
        """Cache regex pattern matches"""
        return f"{text}_{pattern_key}"
    
    def cache_prediction(self, text: str, model_name: str, prediction: Any):
        """Cache model prediction"""
        cache_key = f"{model_name}_{self.get_text_hash(text)}"
        self.prediction_cache.put(cache_key, prediction)
    
    def get_cached_prediction(self, text: str, model_name: str) -> Optional[Any]:
        """Retrieve cached prediction"""
        cache_key = f"{model_name}_{self.get_text_hash(text)}"
        return self.prediction_cache.get(cache_key)

# Global cache instance
model_cache = ModelCache()
performance_monitor = PerformanceMonitor()

def cached_prediction(model_name: str):
    """Decorator for caching model predictions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, text: str, *args, **kwargs):
            # Check cache first
            cached_result = model_cache.get_cached_prediction(text, model_name)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(self, text, *args, **kwargs)
            
            # Cache result
            model_cache.cache_prediction(text, model_name, result)
            
            return result
        return wrapper
    return decorator

def memory_efficient_batch_processing(items: List[Any], batch_size: int = 32):
    """Process items in memory-efficient batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
        # Force garbage collection between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def optimize_model_memory(model):
    """Optimize model for memory efficiency"""
    if hasattr(model, 'half'):
        model = model.half()  # Use FP16
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    return model
