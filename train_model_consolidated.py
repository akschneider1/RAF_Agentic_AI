
#!/usr/bin/env python3
"""
Consolidated, memory-optimized training pipeline for Arabic PII detection
Combines all training approaches into one clean, efficient system
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import pickle
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import pandas as pd
from preprocessing import NERPreprocessor
from performance_optimizer import performance_monitor, log_memory_usage, optimize_model_memory

class ConsolidatedPIITrainer:
    """Memory-optimized, comprehensive PII trainer"""
    
    def __init__(self, model_name: str = 'aubmindlab/bert-base-arabertv2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = NERPreprocessor(model_name)
        self.model = None
        
        # Memory tracking
        log_memory_usage("trainer_init")
        
    def prepare_datasets(self, max_length: int = 128) -> Dict[str, Dataset]:
        """Load and preprocess datasets with memory optimization"""
        print("🔄 PREPARING DATASETS")
        print("=" * 50)
        
        datasets = {}
        
        # Load data files
        train_df = self._load_data_file('Wojood/Wojood1_1_nested/train.csv')
        val_df = self._load_data_file('Wojood/Wojood1_1_nested/val.csv')
        test_df = self._load_data_file('Wojood/Wojood1_1_nested/test.csv')
        
        # Preprocess with memory optimization
        print("Processing training data...")
        train_examples = self.preprocessor.preprocess_dataset(train_df, max_length=max_length)
        log_memory_usage("after_train_processing")
        
        print("Processing validation data...")
        val_examples = self.preprocessor.preprocess_dataset(val_df, max_length=max_length)
        log_memory_usage("after_val_processing")
        
        print("Processing test data...")
        test_examples = self.preprocessor.preprocess_dataset(test_df, max_length=max_length)
        log_memory_usage("after_test_processing")
        
        # Convert to datasets
        datasets['train'] = self._examples_to_dataset(train_examples)
        datasets['validation'] = self._examples_to_dataset(val_examples)
        datasets['test'] = self._examples_to_dataset(test_examples)
        
        # Clean up memory
        del train_examples, val_examples, test_examples
        optimize_model_memory()
        
        print(f"Dataset sizes:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} examples")
        
        return datasets
    
    def _load_data_file(self, filepath: str) -> pd.DataFrame:
        """Load and validate data file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Standardize column names
        if 'global_sentence_id' in df.columns and 'sentence_id' not in df.columns:
            df['sentence_id'] = df['global_sentence_id']
        
        print(f"Loaded {filepath}: {len(df)} rows, {len(df['sentence_id'].unique())} sentences")
        return df
    
    def _examples_to_dataset(self, examples) -> Dataset:
        """Convert processed examples to HuggingFace dataset"""
        return Dataset.from_dict({
            'input_ids': [ex.input_ids for ex in examples],
            'attention_mask': [ex.attention_mask for ex in examples],
            'labels': [ex.labels for ex in examples]
        })
    
    def initialize_model(self):
        """Initialize model with proper configuration"""
        print("🤖 INITIALIZING MODEL")
        print("=" * 50)
        
        num_labels = len(self.preprocessor.label_to_id)
        print(f"Number of labels: {num_labels}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.preprocessor.aligner.id_to_label,
            label2id=self.preprocessor.label_to_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        log_memory_usage("after_model_init")
        print(f"Model initialized: {self.model_name}")
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            pred_list = []
            label_list = []
            for p, l in zip(prediction, label):
                if l != -100:
                    pred_list.append(self.preprocessor.aligner.id_to_label[p])
                    label_list.append(self.preprocessor.aligner.id_to_label[l])
            true_predictions.append(pred_list)
            true_labels.append(label_list)
        
        # Calculate metrics using seqeval
        results = {
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2),
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
        }
        
        return results
    
    def create_training_arguments(self, output_dir: str = "./model_checkpoints") -> TrainingArguments:
        """Create optimized training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,  # Reduced for faster training
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            push_to_hub=False,
            prediction_loss_only=False,
            warmup_steps=100,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            gradient_checkpointing=True,
            dataloader_drop_last=False,
        )
    
    @performance_monitor.time_function("model_training")
    def train(self):
        """Main training function with comprehensive monitoring"""
        print("🚀 STARTING CONSOLIDATED PII TRAINING")
        print("=" * 60)
        
        try:
            # Prepare datasets
            datasets = self.prepare_datasets(max_length=128)
            
            # Initialize model
            self.initialize_model()
            
            # Create data collator
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Create training arguments
            training_args = self.create_training_arguments()
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets['validation'],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Print training info
            self._print_training_info(datasets, training_args)
            
            # Start training
            print("\n🔥 STARTING TRAINING...")
            print("=" * 60)
            
            trainer.train()
            
            # Evaluate on test set
            print("\n📈 EVALUATING ON TEST SET...")
            test_results = trainer.evaluate(eval_dataset=datasets['test'])
            
            self._print_final_results(test_results)
            
            # Save model
            final_model_path = "./pii_model_final"
            self._save_model(trainer, final_model_path)
            
            return trainer, test_results
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
        finally:
            # Clean up memory
            optimize_model_memory()
    
    def _print_training_info(self, datasets, training_args):
        """Print comprehensive training information"""
        print(f"\n📊 TRAINING CONFIGURATION")
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['validation'])}")
        print(f"Test samples: {len(datasets['test'])}")
        print(f"Number of labels: {len(self.preprocessor.label_to_id)}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Max sequence length: 128")
        
        print(f"\n🏷️ LABEL MAPPINGS:")
        for label, idx in sorted(self.preprocessor.label_to_id.items(), key=lambda x: x[1]):
            print(f"  {idx:2d}: {label}")
    
    def _print_final_results(self, test_results):
        """Print final test results"""
        print("\n🎯 FINAL TEST RESULTS:")
        for key, value in test_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                print(f"  {metric_name}: {value:.4f}")
    
    def _save_model(self, trainer, model_path):
        """Save model and configuration"""
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save label mappings and config
        with open(f"{model_path}/training_config.pkl", 'wb') as f:
            pickle.dump({
                'label_to_id': self.preprocessor.label_to_id,
                'id_to_label': self.preprocessor.aligner.id_to_label,
                'model_name': self.model_name
            }, f)
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"Model saved to: {model_path}")
        print("=" * 60)

def main():
    """Main training function"""
    print("🤖 CONSOLIDATED ARABIC PII DETECTION TRAINING")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    trainer = ConsolidatedPIITrainer()
    
    # Run training
    model, results = trainer.train()
    
    if model and results:
        print("\n🎉 Training completed successfully!")
        
        # Print performance stats
        stats = performance_monitor.get_stats()
        print(f"\n📊 PERFORMANCE STATS:")
        print(f"  Training calls: {stats['counters'].get('model_training_calls', 0)}")
        if 'model_training_execution_time' in stats['metrics']:
            training_time = stats['metrics']['model_training_execution_time']['latest']
            print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final memory usage: {stats['system']['memory_used_mb']:.1f} MB")
    else:
        print("\n💥 Training failed. Check logs above.")

if __name__ == "__main__":
    main()
