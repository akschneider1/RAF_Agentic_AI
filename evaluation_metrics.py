
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class PIIEvaluationSuite:
    """Comprehensive evaluation metrics for PII detection"""
    
    def __init__(self):
        self.pii_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']
    
    def evaluate_token_level(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """Token-level evaluation using seqeval"""
        
        # Overall metrics
        overall_f1 = f1_score(y_true, y_pred)
        overall_precision = precision_score(y_true, y_pred)
        overall_recall = recall_score(y_true, y_pred)
        
        # Detailed classification report
        detailed_report = seq_classification_report(y_true, y_pred, output_dict=True)
        
        return {
            "overall": {
                "f1": overall_f1,
                "precision": overall_precision,
                "recall": overall_recall
            },
            "detailed": detailed_report
        }
    
    def evaluate_entity_level(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """Entity-level evaluation (exact match)"""
        
        def extract_entities(labels: List[str]) -> List[Tuple[int, int, str]]:
            entities = []
            current_entity = None
            start_pos = None
            
            for i, label in enumerate(labels):
                if label.startswith('B-'):
                    if current_entity:
                        entities.append((start_pos, i-1, current_entity))
                    current_entity = label[2:]
                    start_pos = i
                elif label.startswith('I-') and current_entity:
                    continue
                else:
                    if current_entity:
                        entities.append((start_pos, i-1, current_entity))
                        current_entity = None
                        start_pos = None
            
            if current_entity:
                entities.append((start_pos, len(labels)-1, current_entity))
            
            return entities
        
        total_true_entities = 0
        total_pred_entities = 0
        total_correct_entities = 0
        
        entity_metrics = {pii_type: {"tp": 0, "fp": 0, "fn": 0} for pii_type in self.pii_types}
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            true_entities = set(extract_entities(true_seq))
            pred_entities = set(extract_entities(pred_seq))
            
            total_true_entities += len(true_entities)
            total_pred_entities += len(pred_entities)
            total_correct_entities += len(true_entities & pred_entities)
            
            # Per-entity-type metrics
            for entity in true_entities:
                entity_type = entity[2]
                if entity_type in entity_metrics:
                    if entity in pred_entities:
                        entity_metrics[entity_type]["tp"] += 1
                    else:
                        entity_metrics[entity_type]["fn"] += 1
            
            for entity in pred_entities:
                entity_type = entity[2]
                if entity_type in entity_metrics and entity not in true_entities:
                    entity_metrics[entity_type]["fp"] += 1
        
        # Calculate overall metrics
        overall_precision = total_correct_entities / total_pred_entities if total_pred_entities > 0 else 0
        overall_recall = total_correct_entities / total_true_entities if total_true_entities > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Calculate per-entity-type metrics
        per_type_metrics = {}
        for entity_type, metrics in entity_metrics.items():
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_type_metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        
        return {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "total_entities": total_true_entities
            },
            "per_type": per_type_metrics
        }
    
    def create_evaluation_report(self, y_true: List[List[str]], y_pred: List[List[str]]) -> str:
        """Create comprehensive evaluation report"""
        
        token_metrics = self.evaluate_token_level(y_true, y_pred)
        entity_metrics = self.evaluate_entity_level(y_true, y_pred)
        
        report = []
        report.append("=" * 80)
        report.append("PII DETECTION EVALUATION REPORT")
        report.append("=" * 80)
        
        # Token-level metrics
        report.append("\n📊 TOKEN-LEVEL METRICS:")
        report.append("-" * 40)
        report.append(f"Overall Precision: {token_metrics['overall']['precision']:.4f}")
        report.append(f"Overall Recall:    {token_metrics['overall']['recall']:.4f}")
        report.append(f"Overall F1-Score:  {token_metrics['overall']['f1']:.4f}")
        
        # Entity-level metrics
        report.append("\n🎯 ENTITY-LEVEL METRICS (Exact Match):")
        report.append("-" * 40)
        report.append(f"Overall Precision: {entity_metrics['overall']['precision']:.4f}")
        report.append(f"Overall Recall:    {entity_metrics['overall']['recall']:.4f}")
        report.append(f"Overall F1-Score:  {entity_metrics['overall']['f1']:.4f}")
        report.append(f"Total Entities:    {entity_metrics['overall']['total_entities']}")
        
        # Per-type breakdown
        report.append("\n🏷️  PER-TYPE PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"{'Type':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report.append("-" * 55)
        
        for pii_type, metrics in entity_metrics['per_type'].items():
            if metrics['support'] > 0:  # Only show types with actual occurrences
                report.append(f"{pii_type:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}")
        
        return "\n".join(report)
    
    def plot_confusion_matrix(self, y_true: List[List[str]], y_pred: List[List[str]], save_path: str = "confusion_matrix.png"):
        """Plot confusion matrix for PII types"""
        
        # Flatten predictions and true labels
        flat_true = [label for seq in y_true for label in seq if label != 'O']
        flat_pred = [label for seq in y_pred for label in seq if label != 'O']
        
        # Get unique labels
        labels = sorted(list(set(flat_true + flat_pred)))
        
        # Create confusion matrix
        cm = confusion_matrix(flat_true, flat_pred, labels=labels)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('PII Detection Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
