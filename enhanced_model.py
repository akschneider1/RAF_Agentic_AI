
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np

class EnhancedPIIModel(nn.Module):
    """Enhanced PII detection model with multiple encoders"""
    
    def __init__(self, model_name: str = 'aubmindlab/bert-base-arabertv2', num_labels: int = 15):
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Primary BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Additional feature layers
        self.contextual_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        
        # Multi-head attention for entity relationships
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head with CRF
        self.classifier = nn.Linear(self.hidden_size // 2, num_labels)
        
        # Optional CRF layer for sequence labeling
        self.use_crf = False  # Can be enabled for better sequence predictions
        
        # Dropout layers
        self.dropout = nn.Dropout(0.3)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply contextual LSTM
        lstm_output, _ = self.contextual_lstm(sequence_output)
        
        # Apply self-attention for entity relationships
        attended_output, _ = self.entity_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Fuse features
        combined_features = torch.cat([lstm_output, attended_output], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Apply dropout
        fused_features = self.dropout(fused_features)
        
        # Get logits
        logits = self.classifier(fused_features)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only compute loss on valid positions
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            outputs["loss"] = loss
        
        return outputs

class EnsemblePIIDetector:
    """Ensemble model combining rule-based and ML approaches"""
    
    def __init__(self, ml_model_path: str, rules_detector):
        self.ml_model = None  # Load from path
        self.rules_detector = rules_detector
        self.weights = {"ml": 0.7, "rules": 0.3}  # Ensemble weights
    
    def predict(self, text: str, min_confidence: float = 0.7) -> List[Dict]:
        """Ensemble prediction combining both approaches"""
        
        # Get rule-based predictions
        rule_matches = self.rules_detector.detect_all_pii(text, min_confidence)
        
        # Get ML predictions (if model is loaded)
        ml_matches = []
        if self.ml_model:
            ml_matches = self._get_ml_predictions(text)
        
        # Combine and weight predictions
        combined_matches = self._combine_predictions(rule_matches, ml_matches)
        
        return combined_matches
    
    def _get_ml_predictions(self, text: str) -> List[Dict]:
        """Get predictions from ML model"""
        # Implementation would depend on loaded model
        pass
    
    def _combine_predictions(self, rule_matches, ml_matches) -> List[Dict]:
        """Combine predictions from both approaches"""
        combined = []
        
        # Process rule-based matches
        for match in rule_matches:
            combined.append({
                "text": match.text,
                "pii_type": match.pii_type,
                "start_pos": match.start_pos,
                "end_pos": match.end_pos,
                "confidence": match.confidence * self.weights["rules"],
                "source": "rules",
                "pattern_name": match.pattern_name
            })
        
        # Add ML matches and resolve conflicts
        # Implementation would handle overlaps and confidence scoring
        
        return combined
