
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

class NERTrainer:
    """Memory-optimized fine-tuning trainer for Arabic NER using AraBERT"""
    
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
    
    def prepare_datasets(self, max_length: int = 256):
        """Load and preprocess datasets with memory optimization"""
        print("PREPARING DATASETS")
        print("=" * 50)
        
        # Load preprocessor if not already loaded
        if self.preprocessor is None:
            self.preprocessor = NERPreprocessor(self.model_name)
            self.label_to_id = self.preprocessor.label_to_id
            self.id_to_label = self.preprocessor.aligner.id_to_label
        
        datasets = {}
        
        # Load original Wojood training data
        print("Loading original Wojood training data...")
        train_df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        # Add sentence_id if not present
        if 'sentence_id' not in train_df.columns:
            train_df['sentence_id'] = train_df['global_sentence_id']
        
        # Load validation data
        val_df = pd.read_csv('Wojood/Wojood1_1_nested/val.csv')
        if 'sentence_id' not in val_df.columns:
            val_df['sentence_id'] = val_df['global_sentence_id']
        
        # Load test data
        test_df = pd.read_csv('Wojood/Wojood1_1_nested/test.csv')
        if 'sentence_id' not in test_df.columns:
            test_df['sentence_id'] = test_df['global_sentence_id']
        
        # Preprocess datasets with shorter sequences
        print("Preprocessing training data...")
        train_examples = self.preprocessor.preprocess_dataset(train_df, max_length=max_length)
        
        print("Preprocessing validation data...")
        val_examples = self.preprocessor.preprocess_dataset(val_df, max_length=max_length)
        
        print("Preprocessing test data...")
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
        
        print(f"Dataset sizes:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} examples")
        
        return datasets
    
    def initialize_model(self):
        """Initialize the model for token classification"""
        print("INITIALIZING MODEL")
        print("=" * 50)
        
        num_labels = len(self.label_to_id)
        print(f"Number of labels: {num_labels}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        
        print(f"Model initialized: {self.model_name}")
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute metrics using seqeval"""
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
        
        # Calculate metrics using seqeval
        results = {
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2),
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
        }
        
        return results
    
    def create_training_arguments(self, output_dir: str = "./model_checkpoints"):
        """Create optimized training arguments with advanced strategies"""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,   # Slightly increased
            per_device_eval_batch_size=8,    
            gradient_accumulation_steps=4,   
            num_train_epochs=5,              # Increased epochs
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=500,                  # More frequent evaluation
            save_strategy="steps",
            save_steps=500,
            logging_strategy="steps",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,              # Keep more checkpoints
            report_to=None,
            seed=42,
            fp16=True,                       # Enable mixed precision for efficiency
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            push_to_hub=False,
            prediction_loss_only=False,
            # Advanced training strategies
            warmup_steps=500,                # Learning rate warmup
            lr_scheduler_type="linear",      # Learning rate scheduling
            optim="adamw_torch",            # Optimizer selection
            gradient_checkpointing=True,     # Memory optimization
            dataloader_drop_last=False,     # Keep all data
            eval_accumulation_steps=1,      # Evaluation efficiency
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
        )
    
    def train(self):
        """Main training function"""
        print("🚀 STARTING ARABERT PII FINE-TUNING")
        print("=" * 60)
        
        # Load configuration
        self.load_preprocessor_config()
        
        # Prepare datasets with shorter sequences
        datasets = self.prepare_datasets(max_length=256)
        
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
            compute_metrics=self.compute_metrics
        )
        
        # Print training info
        print(f"\n📊 MEMORY-OPTIMIZED TRAINING CONFIGURATION")
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['validation'])}")
        print(f"Test samples: {len(datasets['test'])}")
        print(f"Number of labels: {len(self.label_to_id)}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Max sequence length: 256")
        
        print(f"\n🏷️ LABEL MAPPINGS:")
        for label, idx in sorted(self.label_to_id.items(), key=lambda x: x[1]):
            print(f"  {idx:2d}: {label}")
        
        # Start training
        print(f"\n🔥 STARTING TRAINING...")
        print("=" * 60)
        
        try:
            trainer.train()
            
            # Evaluate on test set
            print("\n📈 EVALUATING ON TEST SET...")
            test_results = trainer.evaluate(eval_dataset=datasets['test'])
            
            print("\n🎯 FINAL TEST RESULTS:")
            for key, value in test_results.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    print(f"  {metric_name}: {value:.4f}")
            
            # Save the final model
            final_model_path = "./arabert_pii_model"
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Save label mappings
            with open(f"{final_model_path}/label_mappings.pkl", 'wb') as f:
                pickle.dump({
                    'label_to_id': self.label_to_id,
                    'id_to_label': self.id_to_label
                }, f)
            
            print(f"\n✅ TRAINING COMPLETED!")
            print(f"Model saved to: {final_model_path}")
            print("=" * 60)
            
            return trainer, test_results
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main function to start training"""
    print("🤖 ARABERT PII DETECTION TRAINING PIPELINE")
    print("=" * 60)
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Force CPU usage for memory optimization
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Initialize trainer
    trainer_obj = NERTrainer()
    
    # Run training
    trainer, test_results = trainer_obj.train()
    
    if trainer and test_results:
        print("\n🎉 ALL DONE! Your AraBERT PII model is ready for inference.")
    else:
        print("\n💥 Training failed. Please check the logs above.")

if __name__ == "__main__":
    # Create preprocessor first if needed
    if not os.path.exists('preprocessor_config.pkl'):
        print("Creating preprocessing pipeline first...")
        create_preprocessing_pipeline()
    
    # Start training
    main()
