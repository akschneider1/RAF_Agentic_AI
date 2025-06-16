
#!/usr/bin/env python3
"""
Centralized configuration management for the PII detection system
"""

import os
from pathlib import Path
from typing import Dict, Any
import json

class Config:
    """Centralized configuration management"""
    
    # Model Configuration
    MODEL_NAME = "aubmindlab/bert-base-arabertv2"
    MUTAZ_MODEL_NAME = "MutazYoune/Arabic-NER-PII"
    MAX_SEQUENCE_LENGTH = 128
    
    # Training Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    MAX_BATCH_SIZE = 100
    
    # Performance Configuration
    CACHE_SIZE = 2000
    ENABLE_CACHING = True
    ENABLE_PERFORMANCE_MONITORING = True
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    WOJOOD_DIR = BASE_DIR / "Wojood"
    MODEL_CHECKPOINTS_DIR = BASE_DIR / "model_checkpoints"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Confidence Thresholds
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    RULE_CONFIDENCE_THRESHOLD = 0.7
    ML_CONFIDENCE_THRESHOLD = 0.6
    
    # Ensemble Weights
    ENSEMBLE_WEIGHTS = {
        'rules': 0.4,
        'mutazyoune': 0.3,
        'wojood': 0.3
    }
    
    @classmethod
    def load_from_file(cls, config_file: str):
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(cls, key, value)
    
    @classmethod
    def save_to_file(cls, config_file: str):
        """Save current configuration to JSON file"""
        config_data = {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

# Initialize configuration
config = Config()

# Create necessary directories
config.MODEL_CHECKPOINTS_DIR.mkdir(exist_ok=True)
config.LOGS_DIR.mkdir(exist_ok=True)
