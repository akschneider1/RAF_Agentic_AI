
#!/usr/bin/env python3
"""
Memory-optimized training script to fix evaluation bottlenecks
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
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import pandas as pd
from preprocessing import NERPreprocessor, create_preprocessing_pipeline

class OptimizedNERTrainer:
    """Memory-optimized trainer to fix evaluation bottlenecks"""
    
    def __init__(self, model_name: str = 'aubmindlab/bert-base-arabertv2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = None
        self.model = None
        self.label_to_id = None
        self.id_to_label = None
        
    def load_preprocessor_config(self, config_path: str = 'preprocessor_config.pkl'):
        """Load preprocessor configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.label_to_id = config['label_to_id']
            self.id_to_label = {v: k for k, v in self.label_to_id.items()}
            print(f"Loaded preprocessor config with {len(self.label_to_id)} labels")
        else:
            print("Creating new preprocessor config...")
            self.preprocessor = NERPreprocessor(self.model_name)
            self.label_to_id = self.preprocessor.label_to_id
            self.id_to_label = self.preprocessor.aligner.id_to_label
    
    def prepare_datasets(self, max_length: int = 128):
        """Load datasets with shorter sequences and reduced validation set"""
        print("PREPARING OPTIMIZED DATASETS")
        print("=" * 50)
        
        if self.preprocessor is None:
            self.preprocessor = NERPreprocessor(self.model_name)
            self.label_to_id = self.preprocessor.label_to_id
            self.id_to_label = self.preprocessor.aligner.id_to_label
        
        datasets = {}
        
        # Load training data
        print("Loading training data...")
        train_df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        if 'sentence_id' not in train_df.columns:
            train_df['sentence_id'] = train_df['global_sentence_id']
        
        # Load validation data and REDUCE SIZE for faster evaluation
        val_df = pd.read_csv('Wojood/Wojood1_1_nested/val.csv')
        if 'sentence_id' not in val_df.columns:
            val_df['sentence_id'] = val_df['global_sentence_id']
        
        # **KEY FIX**: Reduce validation set to 20% for faster evaluation
        unique_val_sentences = val_df['sentence_id'].unique()
        selected_val_sentences = unique_val_sentences[:len(unique_val_sentences)//5]  # Take only 20%
        val_df = val_df[val_df['sentence_id'].isin(selected_val_sentences)]
        print(f"Reduced validation set to {len(selected_val_sentences)} sentences for faster evaluation")
        
        # Load test data
        test_df = pd.read_csv('Wojood/Wojood1_1_nested/test.csv')
        if 'sentence_id' not in test_df.columns:
            test_df['sentence_id'] = test_df['global_sentence_id']
        
        # Preprocess with shorter sequences
        print("Preprocessing with optimized settings...")
        train_examples = self.preprocessor.preprocess_dataset(train_df, max_length=max_length)
        val_examples = self.preprocessor.preprocess_dataset(val_df, max_length=max_length)
        test_examples = self.preprocessor.preprocess_dataset(test_df, max_length=max_length)
        
        # Convert to HuggingFace datasets
        def examples_to_dataset(examples):
            return Dataset.from_dict({
                'input_ids': [ex.input_ids for ex in examples],
                'attention_mask': [ex.attention_mask for ex in examples],
                'labels': [ex.labels for ex in examples]
            })
        
        datasets['train'] = examples_to_dataset(train_examples)
        datasets['validation'] = examples_to_dataset(val_examples)
        datasets['test'] = examples_to_dataset(test_examples)
        
        print(f"Optimized dataset sizes:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} examples")
        
        return datasets
    
    def initialize_model(self):
        """Initialize model with memory optimizations"""
        print("INITIALIZING OPTIMIZED MODEL")
        print("=" * 50)
        
        num_labels = len(self.label_to_id)
        print(f"Number of labels: {num_labels}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print(f"Model initialized with memory optimizations: {self.model_name}")
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Fast metrics computation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Calculate only essential metrics for speed
        results = {
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2),
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
        }
        
        return results
    
    def create_optimized_training_arguments(self, output_dir: str = "./optimized_checkpoints"):
        """Create memory-optimized training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-5,
            per_device_train_batch_size=4,      # **REDUCED** from 8
            per_device_eval_batch_size=2,       # **REDUCED** from 8 
            gradient_accumulation_steps=8,      # **INCREASED** to maintain effective batch size
            num_train_epochs=3,                 # **REDUCED** epochs for faster completion
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=1000,                    # **LESS FREQUENT** evaluation
            save_strategy="steps",
            save_steps=1000,
            logging_strategy="steps",
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,                 # **REDUCED** checkpoints to save memory
            report_to=None,
            seed=42,
            fp16=True,                          # **ENABLED** for memory savings
            dataloader_pin_memory=False,        # **DISABLED** to save memory
            dataloader_num_workers=0,           # **SINGLE THREADED** for stability
            remove_unused_columns=False,
            push_to_hub=False,
            prediction_loss_only=False,
            warmup_steps=200,                   # **REDUCED** warmup
            lr_scheduler_type="linear",
            optim="adamw_torch",
            gradient_checkpointing=True,        # **ENABLED** for memory savings
            dataloader_drop_last=True,          # **DROP** incomplete batches
            eval_accumulation_steps=4,          # **ACCUMULATE** eval steps to reduce memory
            # **MEMORY OPTIMIZATIONS**
            max_grad_norm=1.0,                  # Gradient clipping
            ignore_data_skip=True,              # Skip data loading optimizations
        )
    
    def resume_from_checkpoint(self):
        """Resume training from the latest checkpoint"""
        print("🔄 RESUMING FROM CHECKPOINT 2000")
        print("=" * 60)
        
        # Load configuration
        self.load_preprocessor_config()
        
        # Prepare optimized datasets
        datasets = self.prepare_datasets(max_length=128)  # Shorter sequences
        
        # Initialize model
        self.initialize_model()
        
        # Create data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=128
        )
        
        # Create optimized training arguments
        training_args = self.create_optimized_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        print(f"\n📊 OPTIMIZED TRAINING CONFIGURATION")
        print(f"Training samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['validation'])} (REDUCED)")
        print(f"Test samples: {len(datasets['test'])}")
        print(f"Max sequence length: 128 (REDUCED)")
        print(f"Train batch size: 4 (REDUCED)")
        print(f"Eval batch size: 2 (REDUCED)")
        print(f"Gradient accumulation: 8 (INCREASED)")
        print(f"Effective batch size: 32")
        
        try:
            # Clear memory before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Resume training from checkpoint
            print(f"\n🚀 RESUMING TRAINING FROM CHECKPOINT...")
            trainer.train(resume_from_checkpoint="./model_checkpoints/checkpoint-2000")
            
            # Quick evaluation on test set
            print("\n📈 FINAL EVALUATION...")
            test_results = trainer.evaluate(eval_dataset=datasets['test'])
            
            print("\n🎯 FINAL RESULTS:")
            for key, value in test_results.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    print(f"  {metric_name}: {value:.4f}")
            
            # Save the final model
            final_model_path = "./arabert_pii_model_optimized"
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Save label mappings
            with open(f"{final_model_path}/label_mappings.pkl", 'wb') as f:
                pickle.dump({
                    'label_to_id': self.label_to_id,
                    'id_to_label': self.id_to_label
                }, f)
            
            print(f"\n✅ OPTIMIZED TRAINING COMPLETED!")
            print(f"Model saved to: {final_model_path}")
            
            return trainer, test_results
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main function with memory cleanup"""
    print("🤖 OPTIMIZED ARABERT PII TRAINING (BOTTLENECK FIX)")
    print("=" * 60)
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize optimized trainer
    trainer_obj = OptimizedNERTrainer()
    
    # Resume from checkpoint with optimizations
    trainer, test_results = trainer_obj.resume_from_checkpoint()
    
    if trainer and test_results:
        print("\n🎉 BOTTLENECK FIXED! Training completed successfully.")
    else:
        print("\n💥 Still having issues. Check logs above.")

if __name__ == "__main__":
    main()
