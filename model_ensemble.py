
#!/usr/bin/env python3
"""
Ensemble PII detection system combining MutazYoune/Arabic-NER-PII 
with our rule-based and Wojood-trained approaches
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from rules import PIIDetector, PIIMatch
import re

@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple models"""
    text: str
    pii_type: str
    start_pos: int
    end_pos: int
    confidence: float
    source_models: List[str]
    individual_scores: Dict[str, float]

class MutazYouneIntegration:
    """Integration wrapper for MutazYoune/Arabic-NER-PII model"""
    
    def __init__(self):
        self.model_name = "MutazYoune/Arabic-NER-PII"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.label_mapping = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the MutazYoune model"""
        try:
            print(f"Loading MutazYoune model: {self.model_name}")
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Create label mapping to our PII types
            self._create_label_mapping()
            
            print("✅ MutazYoune model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load MutazYoune model: {e}")
            self.pipeline = None
    
    def _create_label_mapping(self):
        """Map MutazYoune labels to our PII taxonomy"""
        # This mapping may need adjustment based on actual MutazYoune labels
        self.label_mapping = {
            # Common NER to PII mappings
            "PERSON": "PERSON",
            "PER": "PERSON",
            "LOCATION": "LOCATION",
            "LOC": "LOCATION",
            "ORGANIZATION": "ORGANIZATION",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",  # Geopolitical entity
            "MISC": "MISC",
            
            # Potential PII-specific mappings
            "PHONE": "PHONE",
            "EMAIL": "EMAIL",
            "ID": "ID_NUMBER",
            "NATIONAL_ID": "ID_NUMBER",
            "ADDRESS": "ADDRESS",
            "DATE": "DATE",
            "TIME": "TIME",
        }
    
    def predict(self, text: str, min_confidence: float = 0.5) -> List[EnsemblePrediction]:
        """Get predictions from MutazYoune model"""
        if not self.pipeline:
            return []
        
        try:
            # Get NER predictions
            ner_results = self.pipeline(text)
            
            predictions = []
            for result in ner_results:
                # Map to our PII taxonomy
                original_label = result['entity_group'].upper()
                mapped_pii_type = self.label_mapping.get(original_label, "MISC")
                
                if result['score'] >= min_confidence:
                    prediction = EnsemblePrediction(
                        text=result['word'],
                        pii_type=mapped_pii_type,
                        start_pos=result['start'],
                        end_pos=result['end'],
                        confidence=result['score'],
                        source_models=['mutazyoune'],
                        individual_scores={'mutazyoune': result['score']}
                    )
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            print(f"Error in MutazYoune prediction: {e}")
            return []

class AdvancedEnsembleDetector:
    """Advanced ensemble PII detector combining multiple approaches"""
    
    def __init__(self):
        # Initialize component detectors
        self.rule_detector = PIIDetector()
        self.mutaz_detector = MutazYouneIntegration()
        
        # Ensemble weights (can be tuned based on validation performance)
        self.model_weights = {
            'rules': 0.4,          # High precision, good for structured PII
            'mutazyoune': 0.3,     # Good for Arabic NER
            'wojood': 0.3          # Our trained model (when available)
        }
        
        # Confidence thresholds per model
        self.confidence_thresholds = {
            'rules': 0.7,
            'mutazyoune': 0.6,
            'wojood': 0.5
        }
    
    def detect_ensemble_pii(self, text: str, min_confidence: float = 0.6) -> List[EnsemblePrediction]:
        """Detect PII using ensemble of all available models with intelligent routing"""
        
        # Quick text analysis for model routing
        text_features = self._analyze_text_features(text)
        selected_models = self._select_optimal_models(text_features)
        
        all_predictions = []
        
        # Get rule-based predictions
        rule_matches = self.rule_detector.detect_all_pii(
            text, 
            min_confidence=self.confidence_thresholds['rules']
        )
        
        # Convert rule matches to ensemble predictions
        for match in rule_matches:
            pred = EnsemblePrediction(
                text=match.text,
                pii_type=match.pii_type,
                start_pos=match.start_pos,
                end_pos=match.end_pos,
                confidence=match.confidence * self.model_weights['rules'],
                source_models=['rules'],
                individual_scores={'rules': match.confidence}
            )
            all_predictions.append(pred)
        
        # Get MutazYoune predictions
        mutaz_predictions = self.mutaz_detector.predict(
            text, 
            min_confidence=self.confidence_thresholds['mutazyoune']
        )
        
        for pred in mutaz_predictions:
            pred.confidence *= self.model_weights['mutazyoune']
            pred.individual_scores['mutazyoune'] *= self.model_weights['mutazyoune']
            all_predictions.append(pred)
        
        # Merge overlapping predictions
        merged_predictions = self._merge_overlapping_predictions(all_predictions)
        
        # Filter by ensemble confidence
        final_predictions = [
            pred for pred in merged_predictions 
            if pred.confidence >= min_confidence
        ]
        
        return final_predictions
    
    def _merge_overlapping_predictions(self, predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Merge overlapping predictions with confidence voting"""
        if not predictions:
            return []
        
        # Sort by start position
        predictions.sort(key=lambda x: x.start_pos)
        
        merged = []
        current_group = [predictions[0]]
        
        for pred in predictions[1:]:
            # Check if overlaps with current group
            overlaps_with_group = any(
                self._predictions_overlap(pred, group_pred) 
                for group_pred in current_group
            )
            
            if overlaps_with_group:
                current_group.append(pred)
            else:
                # Process current group and start new one
                merged_pred = self._merge_prediction_group(current_group)
                if merged_pred:
                    merged.append(merged_pred)
                current_group = [pred]
        
        # Process final group
        if current_group:
            merged_pred = self._merge_prediction_group(current_group)
            if merged_pred:
                merged.append(merged_pred)
        
        return merged
    
    def _predictions_overlap(self, pred1: EnsemblePrediction, pred2: EnsemblePrediction) -> bool:
        """Check if two predictions overlap"""
        return (pred1.start_pos < pred2.end_pos and pred1.end_pos > pred2.start_pos)
    
    def _merge_prediction_group(self, group: List[EnsemblePrediction]) -> Optional[EnsemblePrediction]:
        """Merge a group of overlapping predictions"""
        if not group:
            return None
        
        if len(group) == 1:
            return group[0]
        
        # Find the prediction with highest confidence
        best_pred = max(group, key=lambda x: x.confidence)
        
        # Combine information from all predictions
        all_sources = []
        all_scores = {}
        total_confidence = 0
        
        for pred in group:
            all_sources.extend(pred.source_models)
            all_scores.update(pred.individual_scores)
            total_confidence += pred.confidence
        
        # Average confidence weighted by number of agreeing models
        ensemble_confidence = total_confidence / len(group)
        
        # Use the bounds of the highest confidence prediction
        merged = EnsemblePrediction(
            text=best_pred.text,
            pii_type=best_pred.pii_type,
            start_pos=best_pred.start_pos,
            end_pos=best_pred.end_pos,
            confidence=ensemble_confidence,
            source_models=list(set(all_sources)),
            individual_scores=all_scores
        )
        
        return merged
    
    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text to determine optimal model selection"""
        features = {
            'has_arabic': bool(re.search(r'[\u0600-\u06FF]', text)),
            'has_english': bool(re.search(r'[a-zA-Z]', text)),
            'has_digits': bool(re.search(r'\d', text)),
            'length': len(text),
            'has_structured_patterns': bool(re.search(r'[\+\-\(\)@\.]', text)),
            'complexity_score': len(text.split()) / max(len(text.split()), 1)
        }
        return features
    
    def _select_optimal_models(self, features: Dict[str, Any]) -> List[str]:
        """Select optimal models based on text features"""
        models = ['rules']  # Always include rules
        
        # Add ML models based on text characteristics
        if features['has_arabic'] and features['length'] > 10:
            models.append('mutazyoune')
        
        if features['has_structured_patterns']:
            # Prioritize rules for structured data
            models = ['rules'] + [m for m in models if m != 'rules']
        
        return models
    
    def batch_detect_pii(self, texts: List[str], min_confidence: float = 0.6) -> List[List[EnsemblePrediction]]:
        """Batch processing for multiple texts"""
        from performance_optimizer import memory_efficient_batch_processing
        
        results = []
        for batch in memory_efficient_batch_processing(texts, batch_size=16):
            batch_results = []
            for text in batch:
                predictions = self.detect_ensemble_pii(text, min_confidence)
                batch_results.append(predictions)
            results.extend(batch_results)
        
        return results

    def analyze_model_agreement(self, text: str) -> Dict[str, Any]:
        """Analyze agreement between different models"""
        
        # Get predictions from each model separately
        rule_matches = self.rule_detector.detect_all_pii(text, 0.5)
        mutaz_predictions = self.mutaz_detector.predict(text, 0.5)
        
        analysis = {
            "text": text,
            "model_counts": {
                "rules": len(rule_matches),
                "mutazyoune": len(mutaz_predictions)
            },
            "agreements": [],
            "disagreements": [],
            "confidence_stats": {}
        }
        
        # Analyze agreements and disagreements
        for rule_match in rule_matches:
            # Find overlapping MutazYoune predictions
            overlapping_mutaz = [
                pred for pred in mutaz_predictions
                if (rule_match.start_pos < pred.end_pos and 
                    rule_match.end_pos > pred.start_pos)
            ]
            
            if overlapping_mutaz:
                analysis["agreements"].append({
                    "text": rule_match.text,
                    "rule_type": rule_match.pii_type,
                    "mutaz_types": [pred.pii_type for pred in overlapping_mutaz],
                    "rule_confidence": rule_match.confidence,
                    "mutaz_confidences": [pred.confidence for pred in overlapping_mutaz]
                })
            else:
                analysis["disagreements"].append({
                    "text": rule_match.text,
                    "only_in": "rules",
                    "type": rule_match.pii_type,
                    "confidence": rule_match.confidence
                })
        
        return analysis

def test_ensemble_detector():
    """Test the ensemble detector with sample texts"""
    
    detector = AdvancedEnsembleDetector()
    
    test_texts = [
        # Arabic PII examples
        "اسمي أحمد محمد وأعمل في شركة أرامكو",
        "رقم هاتفي 0501234567 وإيميلي ahmed@saudi.com",
        "رقم الهوية الوطنية: 1234567890",
        "العنوان: الرياض، حي النخيل، شارع الملك فهد",
        
        # Mixed Arabic-English
        "Contact Dr. أحمد المحمد at +966501234567 or ahmad@ksu.edu.sa",
        "شركة Google في المملكة العربية السعودية",
        
        # English PII
        "My name is John Smith and my phone is +1-555-123-4567",
        "Email me at john.smith@company.com for more information"
    ]
    
    print("🔬 TESTING ENSEMBLE PII DETECTOR")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 50)
        
        # Get ensemble predictions
        predictions = detector.detect_ensemble_pii(text, min_confidence=0.5)
        
        if predictions:
            for pred in predictions:
                sources = ", ".join(pred.source_models)
                print(f"🎯 {pred.pii_type}: '{pred.text}' "
                      f"(conf: {pred.confidence:.3f}, sources: {sources})")
        else:
            print("No PII detected")
        
        # Analyze model agreement
        agreement = detector.analyze_model_agreement(text)
        if agreement["agreements"] or agreement["disagreements"]:
            print(f"📊 Agreement Analysis:")
            print(f"   Rules found: {agreement['model_counts']['rules']}")
            print(f"   MutazYoune found: {agreement['model_counts']['mutazyoune']}")
            print(f"   Agreements: {len(agreement['agreements'])}")
            print(f"   Disagreements: {len(agreement['disagreements'])}")

if __name__ == "__main__":
    test_ensemble_detector()
