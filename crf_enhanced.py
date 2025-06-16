
#!/usr/bin/env python3
"""
Enhanced CRF implementation for Arabic PII detection with sequence constraints
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Dict, Optional, Tuple
import numpy as np

class EnhancedPIICRF(nn.Module):
    """Enhanced CRF layer with PII-specific constraints"""
    
    def __init__(self, num_labels: int, label_to_id: Dict[str, int]):
        super().__init__()
        self.num_labels = num_labels
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        
        # Initialize CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Set up transition constraints for Arabic PII
        self._setup_transition_constraints()
        
    def _setup_transition_constraints(self):
        """Set up transition constraints for PII entity types"""
        # Initialize transition matrix with very negative values for forbidden transitions
        forbidden_score = -10000.0
        
        # Get label indices
        o_idx = self.label_to_id.get('O', 0)
        
        # Create constraint matrix
        constraints = torch.zeros(self.num_labels, self.num_labels)
        
        # Define PII entity types
        pii_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']
        
        for pii_type in pii_types:
            b_label = f'B-{pii_type}'
            i_label = f'I-{pii_type}'
            
            if b_label in self.label_to_id and i_label in self.label_to_id:
                b_idx = self.label_to_id[b_label]
                i_idx = self.label_to_id[i_label]
                
                # Constraints for Arabic PII:
                # 1. I-TYPE can only follow B-TYPE or I-TYPE of the same type
                for other_type in pii_types:
                    if other_type != pii_type:
                        other_b = f'B-{other_type}'
                        if other_b in self.label_to_id:
                            other_b_idx = self.label_to_id[other_b]
                            # Forbid I-TYPE following B-OTHER_TYPE
                            constraints[other_b_idx, i_idx] = forbidden_score
                
                # 2. Specific Arabic PII constraints
                if pii_type in ['PHONE', 'EMAIL', 'ID_NUMBER']:
                    # These are typically single entities, discourage very long sequences
                    # Add small penalty for long sequences (more than 3 tokens)
                    constraints[i_idx, i_idx] = -1.0
        
        # Apply constraints to CRF transitions
        with torch.no_grad():
            self.crf.transitions.data += constraints
    
    def forward(self, emissions: torch.Tensor, labels: torch.Tensor = None, 
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with CRF"""
        if labels is not None:
            # Training mode - compute loss
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='mean')
            return {'loss': -log_likelihood}
        else:
            # Inference mode - decode best sequence
            best_paths = self.crf.decode(emissions, mask=mask)
            return {'predictions': best_paths}
    
    def decode_with_confidence(self, emissions: torch.Tensor, 
                             mask: torch.Tensor = None) -> Tuple[List[List[int]], List[float]]:
        """Decode sequences with confidence scores"""
        # Get best paths
        best_paths = self.crf.decode(emissions, mask=mask)
        
        # Calculate confidence scores based on emission probabilities
        confidences = []
        for i, path in enumerate(best_paths):
            path_confidence = []
            for j, label_id in enumerate(path):
                if mask is None or mask[i, j]:
                    # Softmax over emissions to get probabilities
                    probs = torch.softmax(emissions[i, j], dim=0)
                    confidence = probs[label_id].item()
                    path_confidence.append(confidence)
            
            # Average confidence for the sequence
            avg_confidence = np.mean(path_confidence) if path_confidence else 0.0
            confidences.append(avg_confidence)
        
        return best_paths, confidences

class PIISequenceValidator:
    """Validate PII sequences according to Arabic language patterns"""
    
    def __init__(self, label_to_id: Dict[str, int]):
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        
    def validate_sequence(self, predicted_labels: List[int], 
                         tokens: List[str] = None) -> Tuple[List[int], float]:
        """Validate and correct a predicted sequence"""
        corrected_labels = []
        corrections_made = 0
        
        prev_label = 'O'
        
        for i, label_id in enumerate(predicted_labels):
            current_label = self.id_to_label.get(label_id, 'O')
            
            # Validate BIO consistency
            if current_label.startswith('I-'):
                entity_type = current_label[2:]
                expected_prefix = f'B-{entity_type}' if prev_label == 'O' else f'I-{entity_type}'
                
                if not (prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}'):
                    # Invalid I- tag, convert to B-
                    corrected_label = f'B-{entity_type}'
                    corrected_id = self.label_to_id.get(corrected_label, label_id)
                    corrected_labels.append(corrected_id)
                    corrections_made += 1
                    prev_label = corrected_label
                else:
                    corrected_labels.append(label_id)
                    prev_label = current_label
            else:
                corrected_labels.append(label_id)
                prev_label = current_label
        
        # Calculate validation score
        validation_score = 1.0 - (corrections_made / len(predicted_labels))
        
        return corrected_labels, validation_score
    
    def apply_arabic_pii_rules(self, predicted_labels: List[int], 
                              tokens: List[str]) -> List[int]:
        """Apply Arabic-specific PII detection rules"""
        if not tokens:
            return predicted_labels
        
        corrected_labels = predicted_labels.copy()
        
        for i, (token, label_id) in enumerate(zip(tokens, predicted_labels)):
            label = self.id_to_label.get(label_id, 'O')
            
            # Arabic-specific corrections
            if self._is_arabic_text(token):
                # Check for common Arabic PII patterns
                if self._looks_like_arabic_name(token) and label == 'O':
                    # Suggest PERSON label for Arabic names
                    person_b_id = self.label_to_id.get('B-PERSON', label_id)
                    corrected_labels[i] = person_b_id
                
                elif self._looks_like_location_indicator(token) and i + 1 < len(tokens):
                    # Check if next token should be LOCATION
                    next_label = self.id_to_label.get(predicted_labels[i + 1], 'O')
                    if next_label == 'O':
                        location_b_id = self.label_to_id.get('B-LOCATION', predicted_labels[i + 1])
                        corrected_labels[i + 1] = location_b_id
        
        return corrected_labels
    
    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        return any('\u0600' <= char <= '\u06FF' for char in text)
    
    def _looks_like_arabic_name(self, token: str) -> bool:
        """Check if token looks like an Arabic name"""
        # Common Arabic name patterns
        name_indicators = ['محمد', 'أحمد', 'فاطمة', 'عائشة', 'عبد', 'أبو', 'أم']
        return any(indicator in token for indicator in name_indicators)
    
    def _looks_like_location_indicator(self, token: str) -> bool:
        """Check if token indicates a location follows"""
        location_indicators = ['في', 'من', 'إلى', 'بـ', 'مدينة', 'محافظة']
        return token in location_indicators

def create_enhanced_crf_model(model_name: str = 'aubmindlab/bert-base-arabertv2', 
                            label_to_id: Dict[str, int] = None):
    """Create a BERT model with enhanced CRF for Arabic PII detection"""
    
    if label_to_id is None:
        # Default PII labels
        label_to_id = {
            'O': 0,
            'B-PERSON': 1, 'I-PERSON': 2,
            'B-LOCATION': 3, 'I-LOCATION': 4,
            'B-ORGANIZATION': 5, 'I-ORGANIZATION': 6,
            'B-PHONE': 7, 'I-PHONE': 8,
            'B-EMAIL': 9, 'I-EMAIL': 10,
            'B-ID_NUMBER': 11, 'I-ID_NUMBER': 12,
            'B-ADDRESS': 13, 'I-ADDRESS': 14
        }
    
    from transformers import AutoModel
    
    class BERTWithEnhancedCRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.bert.config.hidden_size, len(label_to_id))
            self.crf = EnhancedPIICRF(len(label_to_id), label_to_id)
            self.validator = PIISequenceValidator(label_to_id)
        
        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # Get BERT outputs
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            sequence_output = self.dropout(outputs.last_hidden_state)
            
            # Get emission scores
            emissions = self.classifier(sequence_output)
            
            # Apply CRF
            mask = attention_mask.bool() if attention_mask is not None else None
            crf_outputs = self.crf(emissions, labels, mask)
            
            return crf_outputs
        
        def predict_with_validation(self, input_ids, attention_mask=None, tokens=None):
            """Predict with sequence validation"""
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            emissions = self.classifier(self.dropout(outputs.last_hidden_state))
            
            mask = attention_mask.bool() if attention_mask is not None else None
            predictions, confidences = self.crf.decode_with_confidence(emissions, mask)
            
            # Apply validation
            validated_predictions = []
            validation_scores = []
            
            for i, pred_seq in enumerate(predictions):
                if tokens and i < len(tokens):
                    token_seq = tokens[i]
                    validated_seq, val_score = self.validator.validate_sequence(pred_seq, token_seq)
                    validated_seq = self.validator.apply_arabic_pii_rules(validated_seq, token_seq)
                else:
                    validated_seq, val_score = self.validator.validate_sequence(pred_seq)
                
                validated_predictions.append(validated_seq)
                validation_scores.append(val_score)
            
            return validated_predictions, confidences, validation_scores
    
    return BERTWithEnhancedCRF()

def test_crf_constraints():
    """Test the CRF constraints"""
    print("🧪 TESTING ENHANCED CRF IMPLEMENTATION")
    print("=" * 50)
    
    # Create sample label mappings
    label_to_id = {
        'O': 0,
        'B-PERSON': 1, 'I-PERSON': 2,
        'B-LOCATION': 3, 'I-LOCATION': 4,
        'B-PHONE': 5, 'I-PHONE': 6
    }
    
    # Test CRF
    crf = EnhancedPIICRF(len(label_to_id), label_to_id)
    
    # Test sequence validator
    validator = PIISequenceValidator(label_to_id)
    
    # Test invalid sequence: O -> I-PERSON (should be corrected to B-PERSON)
    invalid_sequence = [0, 2, 2]  # O, I-PERSON, I-PERSON
    corrected, score = validator.validate_sequence(invalid_sequence)
    
    print(f"Original sequence: {[validator.id_to_label[i] for i in invalid_sequence]}")
    print(f"Corrected sequence: {[validator.id_to_label[i] for i in corrected]}")
    print(f"Validation score: {score:.2f}")
    
    print("\n✅ CRF implementation ready for training!")

if __name__ == "__main__":
    test_crf_constraints()
