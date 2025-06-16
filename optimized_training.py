
#!/usr/bin/env python3
"""
Optimized training pipeline with memory management and dynamic batching
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import numpy as np
from typing import List, Dict, Any
import gc
from performance_optimizer import memory_efficient_batch_processing, optimize_model_memory

class OptimizedPIIDataset(Dataset):
    """Memory-optimized dataset with lazy loading"""
    
    def __init__(self, encodings, labels, max_length=128):
        self.encodings = encodings
        self.labels = labels
        self.max_length = max_length
    
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx][:self.max_length]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx][:self.max_length]),
            'labels': torch.tensor(self.labels[idx][:self.max_length])
        }
        return item
    
    def __len__(self):
        return len(self.labels)

class OptimizedTrainer:
    """Optimized trainer with memory management"""
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def setup_model(self, num_labels: int):
        """Setup model with optimizations"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply memory optimizations
        self.model = optimize_model_memory(self.model)
        self.model.to(self.device)
    
    def create_dynamic_dataloader(self, dataset, batch_size=16):
        """Create dataloader with dynamic batching"""
        def collate_fn(batch):
            # Dynamic padding to max length in batch
            max_len = max(len(item['input_ids']) for item in batch)
            
            input_ids = []
            attention_masks = []
            labels = []
            
            for item in batch:
                input_id = item['input_ids']
                attention_mask = item['attention_mask']
                label = item['labels']
                
                # Pad to max length in batch
                padding_length = max_len - len(input_id)
                
                input_ids.append(torch.cat([
                    input_id, 
                    torch.zeros(padding_length, dtype=torch.long)
                ]))
                attention_masks.append(torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ]))
                labels.append(torch.cat([
                    label,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ]))
            
            return {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_masks),
                'labels': torch.stack(labels)
            }
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def train_with_memory_optimization(self, train_dataset, val_dataset, output_dir="./optimized_model"):
        """Train with aggressive memory optimization"""
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Smaller batch size
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,  # Effective batch size: 32
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Mixed precision
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None  # Disable wandb/tensorboard
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train with memory cleanup
        try:
            trainer.train()
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return trainer
