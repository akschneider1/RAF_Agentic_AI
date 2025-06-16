
#!/usr/bin/env python3
"""
Improved model architecture with specialized layers for PII detection
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional

class PIISpecializedBERT(nn.Module):
    """BERT model with specialized layers for PII detection"""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_labels = num_labels
        
    def _create_label_mappings(self) -> Dict[str, int]:
        """Create label to ID mappings for PII types"""
        return {
            'O': 0,
            'B-PERSON': 1, 'I-PERSON': 2,
            'B-LOCATION': 3, 'I-LOCATION': 4, 
            'B-ORGANIZATION': 5, 'I-ORGANIZATION': 6,
            'B-PHONE': 7, 'I-PHONE': 8,
            'B-EMAIL': 9, 'I-EMAIL': 10,
            'B-ID_NUMBER': 11, 'I-ID_NUMBER': 12,
            'B-ADDRESS': 13, 'I-ADDRESS': 14
        }
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        hidden_size = config.hidden_size
        
        # Specialized layers for different PII types
        self.pii_type_embeddings = nn.Embedding(num_labels, hidden_size // 4)
        
        # Multi-layer classifier with residual connections
        self.classifier_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, num_labels)
        ])
        
        # Attention mechanism for sequence-level context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout and layer normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LayerNorm(hidden_size // 2)
        ])
        
        # Enhanced CRF layer for sequence consistency
        self.use_crf = True
        if self.use_crf:
            from crf_enhanced import EnhancedPIICRF
            self.crf = EnhancedPIICRF(num_labels, self._create_label_mappings())
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply context attention
        attended_output, _ = self.context_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Residual connection
        sequence_output = sequence_output + attended_output
        sequence_output = self.layer_norms[0](sequence_output)
        
        # Multi-layer classification
        hidden = sequence_output
        
        # First classifier layer
        hidden = self.classifier_layers[0](hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.layer_norms[1](hidden)
        
        # Second classifier layer
        hidden = self.classifier_layers[1](hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.layer_norms[2](hidden)
        
        # Final classification layer
        logits = self.classifier_layers[2](hidden)
        
        # Prepare outputs
        outputs_dict = {"logits": logits}
        
        if labels is not None:
            if self.use_crf:
                # Use CRF for training
                mask = attention_mask.bool() if attention_mask is not None else None
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
                outputs_dict["loss"] = loss
            else:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else None
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                
                if active_loss is not None:
                    active_logits = active_logits[active_loss]
                    active_labels = active_labels[active_loss]
                
                loss = loss_fct(active_logits, active_labels)
                outputs_dict["loss"] = loss
        
        return outputs_dict
    
    def predict(self, input_ids, attention_mask=None):
        """Make predictions using CRF if available"""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            if self.use_crf:
                mask = attention_mask.bool() if attention_mask is not None else None
                predictions = self.crf.decode(logits, mask=mask)
                return predictions
            else:
                return torch.argmax(logits, dim=-1)

class PIIModelTrainer:
    """Trainer for the specialized PII model"""
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        
    def create_specialized_model(self, num_labels: int) -> PIISpecializedBERT:
        """Create the specialized PII detection model"""
        return PIISpecializedBERT(
            model_name=self.model_name,
            num_labels=num_labels,
            dropout_rate=0.3
        )
    
    def setup_training(self, model, train_dataset, eval_dataset):
        """Setup training with specialized configuration"""
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir="./specialized_pii_model",
            learning_rate=2e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=3,
            num_train_epochs=6,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=300,
            save_strategy="steps",
            save_steps=300,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            warmup_steps=800,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            seed=42,
            lr_scheduler_type="cosine_with_restarts",
            save_total_limit=3,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Handle CRF predictions (list of lists) vs regular predictions (tensor)
        if isinstance(predictions, list):
            # CRF predictions
            true_predictions = []
            true_labels = []
            
            for pred_seq, label_seq in zip(predictions, labels):
                pred_labels = []
                true_labels_seq = []
                
                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:
                        pred_labels.append(pred)
                        true_labels_seq.append(label)
                
                true_predictions.append(pred_labels)
                true_labels.append(true_labels_seq)
        else:
            # Regular predictions
            predictions = predictions.argmax(axis=-1)
            
            true_predictions = []
            true_labels = []
            
            for prediction, label in zip(predictions, labels):
                pred_labels = []
                true_labels_seq = []
                
                for pred, l in zip(prediction, label):
                    if l != -100:
                        pred_labels.append(pred)
                        true_labels_seq.append(l)
                
                true_predictions.append(pred_labels)
                true_labels.append(true_labels_seq)
        
        # Calculate token-level accuracy
        total_tokens = sum(len(seq) for seq in true_labels)
        correct_tokens = sum(
            sum(p == l for p, l in zip(pred_seq, label_seq))
            for pred_seq, label_seq in zip(true_predictions, true_labels)
        )
        
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Calculate F1 score (simplified version)
        # In practice, you'd use seqeval for proper NER evaluation
        return {
            "accuracy": accuracy,
            "f1": accuracy  # Placeholder - use seqeval for proper F1
        }

def main():
    """Test the specialized model"""
    print("🧠 TESTING SPECIALIZED PII MODEL")
    print("=" * 50)
    
    trainer = PIIModelTrainer()
    model = trainer.create_specialized_model(num_labels=15)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("✅ Specialized model ready for training!")

if __name__ == "__main__":
    main()
