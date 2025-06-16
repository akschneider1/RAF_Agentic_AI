
#!/usr/bin/env python3
"""
Enhanced training pipeline with advanced techniques for PII detection
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
from preprocessing import NERPreprocessor
from typing import List, Dict, Any
import pandas as pd

class AdvancedPIITrainer:
    """Enhanced trainer with class balancing and advanced techniques"""
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = NERPreprocessor(model_name)
        self.model = None
        self.class_weights = None
        
    def calculate_class_weights(self, labels: List[List[int]]) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        # Flatten labels and remove -100 (ignored labels)
        flat_labels = []
        for label_seq in labels:
            flat_labels.extend([l for l in label_seq if l != -100])
        
        # Count label frequencies
        label_counts = Counter(flat_labels)
        
        # Calculate weights inversely proportional to frequencies
        total_samples = len(flat_labels)
        num_classes = len(self.preprocessor.label_to_id)
        
        weights = torch.ones(num_classes)
        for label_id, count in label_counts.items():
            if label_id < num_classes:
                weights[label_id] = total_samples / (num_classes * count)
        
        # Normalize weights
        weights = weights / weights.sum() * num_classes
        
        print(f"Class weights calculated:")
        for label_id, weight in enumerate(weights):
            label_name = self.preprocessor.aligner.id_to_label.get(label_id, f"UNK_{label_id}")
            print(f"  {label_name}: {weight:.3f}")
        
        return weights
    
    def create_focal_loss_trainer(self, train_dataset, eval_dataset, class_weights):
        """Create trainer with focal loss for handling class imbalance"""
        
        class FocalLossTrainer(Trainer):
            def __init__(self, class_weights, alpha=1.0, gamma=2.0, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights.to(self.model.device)
                self.alpha = alpha
                self.gamma = gamma
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Reshape for loss calculation
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    ignore_index=-100,
                    reduction='none'
                )
                
                # Flatten
                active_loss = inputs["attention_mask"].view(-1) == 1
                active_logits = logits.view(-1, model.config.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(-100).type_as(labels)
                )
                
                # Calculate focal loss
                ce_loss = loss_fct(active_logits, active_labels)
                
                # Get probabilities for focal loss
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                # Only consider non-ignored tokens
                valid_loss_mask = (active_labels != -100)
                if valid_loss_mask.sum() > 0:
                    loss = focal_loss[valid_loss_mask].mean()
                else:
                    loss = torch.tensor(0.0, requires_grad=True)
                
                return (loss, outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir="./enhanced_model_checkpoints",
            learning_rate=1e-5,  # Lower learning rate for fine-tuning
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=8,  # More epochs with early stopping
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=5,
            warmup_steps=1000,  # More warmup steps
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            seed=42,
            # Advanced optimization
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            gradient_checkpointing=True,
            # Early stopping
            load_best_model_at_end=True,
        )
        
        trainer = FocalLossTrainer(
            class_weights=class_weights,
            alpha=1.0,
            gamma=2.0,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_enhanced_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        return trainer
    
    def compute_enhanced_metrics(self, eval_pred):
        """Enhanced metrics with per-class performance"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index
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
        
        # Calculate metrics
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        from seqeval.scheme import IOB2
        
        results = {
            'precision': precision_score(true_labels, true_predictions, scheme=IOB2),
            'recall': recall_score(true_labels, true_predictions, scheme=IOB2),
            'f1': f1_score(true_labels, true_predictions, scheme=IOB2),
        }
        
        # Per-entity metrics
        report = classification_report(true_labels, true_predictions, scheme=IOB2, output_dict=True)
        
        # Add per-entity F1 scores
        for entity_type in ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']:
            if entity_type in report:
                results[f'f1_{entity_type}'] = report[entity_type]['f1-score']
        
        return results
    
    def prepare_enhanced_dataset(self):
        """Prepare dataset with quality filtering and augmentation"""
        print("PREPARING ENHANCED DATASET")
        print("=" * 50)
        
        # Load multiple data sources
        datasets = {}
        
        # 1. Original Wojood data
        print("Loading Wojood training data...")
        wojood_train = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        if 'sentence_id' not in wojood_train.columns:
            wojood_train['sentence_id'] = wojood_train['global_sentence_id']
        
        # 2. Augmented data if available
        if os.path.exists('train_augmented.csv'):
            print("Loading augmented training data...")
            augmented_data = pd.read_csv('train_augmented.csv')
            # Combine datasets
            combined_train = pd.concat([wojood_train, augmented_data], ignore_index=True)
        else:
            combined_train = wojood_train
        
        # Quality filtering
        print("Applying quality filters...")
        
        # Remove very short sequences (< 3 tokens)
        sentence_lengths = combined_train.groupby('sentence_id').size()
        valid_sentences = sentence_lengths[sentence_lengths >= 3].index
        combined_train = combined_train[combined_train['sentence_id'].isin(valid_sentences)]
        
        # Remove sequences with too many entities (likely annotation errors)
        entity_ratios = combined_train.groupby('sentence_id').apply(
            lambda x: (x['tags'] != 'O').sum() / len(x)
        )
        valid_sentences = entity_ratios[entity_ratios <= 0.7].index  # Max 70% entities
        combined_train = combined_train[combined_train['sentence_id'].isin(valid_sentences)]
        
        print(f"After quality filtering: {len(combined_train)} tokens in {len(combined_train['sentence_id'].unique())} sentences")
        
        # Preprocess datasets
        train_examples = self.preprocessor.preprocess_dataset(combined_train, max_length=128)
        
        # Load validation and test data
        val_df = pd.read_csv('Wojood/Wojood1_1_nested/val.csv')
        if 'sentence_id' not in val_df.columns:
            val_df['sentence_id'] = val_df['global_sentence_id']
        val_examples = self.preprocessor.preprocess_dataset(val_df, max_length=128)
        
        test_df = pd.read_csv('Wojood/Wojood1_1_nested/test.csv')
        if 'sentence_id' not in test_df.columns:
            test_df['sentence_id'] = test_df['global_sentence_id']
        test_examples = self.preprocessor.preprocess_dataset(test_df, max_length=128)
        
        # Convert to torch datasets
        from torch.utils.data import Dataset
        
        class PIIDataset(Dataset):
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                ex = self.examples[idx]
                return {
                    'input_ids': torch.tensor(ex.input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(ex.attention_mask, dtype=torch.long),
                    'labels': torch.tensor(ex.labels, dtype=torch.long)
                }
        
        datasets['train'] = PIIDataset(train_examples)
        datasets['validation'] = PIIDataset(val_examples)
        datasets['test'] = PIIDataset(test_examples)
        
        # Calculate class weights from training data
        train_labels = [ex.labels for ex in train_examples]
        self.class_weights = self.calculate_class_weights(train_labels)
        
        print(f"Dataset sizes:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} examples")
        
        return datasets
    
    def train_enhanced_model(self):
        """Train model with all enhancements"""
        print("🚀 ENHANCED PII MODEL TRAINING")
        print("=" * 60)
        
        # Prepare datasets
        datasets = self.prepare_enhanced_dataset()
        
        # Initialize model
        num_labels = len(self.preprocessor.label_to_id)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.preprocessor.aligner.id_to_label,
            label2id=self.preprocessor.label_to_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Create enhanced trainer with focal loss
        trainer = self.create_focal_loss_trainer(
            datasets['train'],
            datasets['validation'],
            self.class_weights
        )
        
        print("🔥 Starting enhanced training...")
        
        try:
            # Train the model
            trainer.train()
            
            # Evaluate on test set
            print("\n📈 EVALUATING ON TEST SET...")
            test_results = trainer.evaluate(eval_dataset=datasets['test'])
            
            print("\n🎯 FINAL TEST RESULTS:")
            for key, value in test_results.items():
                if key.startswith('eval_'):
                    metric_name = key.replace('eval_', '')
                    print(f"  {metric_name}: {value:.4f}")
            
            # Save model
            final_model_path = "./enhanced_pii_model"
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Save additional config
            with open(f"{final_model_path}/training_config.pkl", 'wb') as f:
                pickle.dump({
                    'label_to_id': self.preprocessor.label_to_id,
                    'id_to_label': self.preprocessor.aligner.id_to_label,
                    'class_weights': self.class_weights,
                    'model_name': self.model_name
                }, f)
            
            print(f"\n✅ ENHANCED TRAINING COMPLETED!")
            print(f"Model saved to: {final_model_path}")
            
            return trainer, test_results
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main training function"""
    print("🤖 ENHANCED ARABIC PII DETECTION TRAINING")
    print("=" * 60)
    
    trainer = AdvancedPIITrainer()
    model, results = trainer.train_enhanced_model()
    
    if model and results:
        print("\n🎉 Enhanced training completed successfully!")
    else:
        print("\n💥 Training failed. Check logs above.")

if __name__ == "__main__":
    import os
    main()
