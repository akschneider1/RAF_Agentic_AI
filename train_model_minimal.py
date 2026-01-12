
#!/usr/bin/env python3
"""
Minimal, memory-optimized training pipeline
Replaces all previous training implementations
"""

import os
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import pandas as pd
from preprocessing import NERPreprocessor

class MinimalPIITrainer:
    """Streamlined PII trainer - replaces all other training files"""
    
    def __init__(self):
        self.model_name = 'aubmindlab/bert-base-arabertv2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.preprocessor = NERPreprocessor(self.model_name)
        self.model = None
    
    def load_data(self):
        """Load minimal required data"""
        try:
            train_df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
            val_df = pd.read_csv('Wojood/Wojood1_1_nested/val.csv')
            return train_df[:1000], val_df[:200]  # Use subset for speed
        except:
            print("⚠️ Wojood data not found. Training skipped.")
            return None, None
    
    def prepare_datasets(self, train_df, val_df):
        """Minimal dataset preparation"""
        train_examples = self.preprocessor.preprocess_dataset(train_df, max_length=64)
        val_examples = self.preprocessor.preprocess_dataset(val_df, max_length=64)
        
        return {
            'train': Dataset.from_dict({
                'input_ids': [ex.input_ids for ex in train_examples],
                'attention_mask': [ex.attention_mask for ex in train_examples],
                'labels': [ex.labels for ex in train_examples]
            }),
            'validation': Dataset.from_dict({
                'input_ids': [ex.input_ids for ex in val_examples],
                'attention_mask': [ex.attention_mask for ex in val_examples],
                'labels': [ex.labels for ex in val_examples]
            })
        }
    
    def compute_metrics(self, eval_pred):
        """Compute F1 score"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
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
        
        return {
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2)
        }
    
    def train(self):
        """Minimal training function"""
        print("🤖 MINIMAL PII TRAINING")
        print("=" * 40)
        
        # Load data
        train_df, val_df = self.load_data()
        if train_df is None:
            return None
        
        # Prepare datasets
        datasets = self.prepare_datasets(train_df, val_df)
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.preprocessor.label_to_id)
        )
        
        # Training arguments - optimized for speed and memory
        training_args = TrainingArguments(
            output_dir="./models",
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            num_train_epochs=1,  # Quick training
            evaluation_strategy="no",
            save_strategy="no",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            remove_unused_columns=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        print(f"Training samples: {len(datasets['train'])}")
        print("Starting quick training...")
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"F1 Score: {results.get('eval_f1', 0):.3f}")
        
        # Save minimal model
        trainer.save_model("./models/pii_model")
        print("✅ Model saved to ./models/pii_model")
        
        return trainer

def main():
    """Quick training entry point"""
    trainer = MinimalPIITrainer()
    trainer.train()

if __name__ == "__main__":
    main()
