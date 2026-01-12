
#!/usr/bin/env python3
"""
Comprehensive training monitoring and evaluation system
"""

import wandb
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

class PIITrainingMonitor:
    """Monitor PII training progress and performance"""
    
    def __init__(self, project_name: str = "arabic-pii-detection"):
        self.project_name = project_name
        self.metrics_history = []
        
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""
        self.metrics_history.append({
            'epoch': epoch,
            **metrics
        })
        
        # Log to wandb if available
        try:
            wandb.log(metrics, step=epoch)
        except:
            pass
    
    def evaluate_pii_performance(self, predictions: List[List[str]], 
                                true_labels: List[List[str]]) -> Dict[str, float]:
        """Comprehensive PII evaluation"""
        
        # Overall metrics
        overall_metrics = {
            'precision': precision_score(true_labels, predictions, scheme=IOB2),
            'recall': recall_score(true_labels, predictions, scheme=IOB2),
            'f1': f1_score(true_labels, predictions, scheme=IOB2)
        }
        
        # Per-entity metrics
        report = classification_report(true_labels, predictions, scheme=IOB2, output_dict=True)
        
        pii_entities = ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']
        
        for entity in pii_entities:
            if entity in report:
                overall_metrics[f'{entity}_precision'] = report[entity]['precision']
                overall_metrics[f'{entity}_recall'] = report[entity]['recall']
                overall_metrics[f'{entity}_f1'] = report[entity]['f1-score']
                overall_metrics[f'{entity}_support'] = report[entity]['support']
        
        return overall_metrics
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # F1 Score progression
        if 'f1' in df.columns:
            axes[0, 0].plot(df['epoch'], df['f1'], 'b-', label='F1 Score')
            axes[0, 0].set_title('F1 Score Progress')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].legend()
        
        # Loss progression
        if 'loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['loss'], 'r-', label='Loss')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
        
        # Per-entity F1 scores
        entity_columns = [col for col in df.columns if col.endswith('_f1')]
        if entity_columns:
            for col in entity_columns:
                entity_name = col.replace('_f1', '')
                axes[1, 0].plot(df['epoch'], df[col], label=entity_name)
            axes[1, 0].set_title('Per-Entity F1 Scores')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
        
        # Precision vs Recall
        if 'precision' in df.columns and 'recall' in df.columns:
            axes[1, 1].scatter(df['recall'], df['precision'], c=df['epoch'], cmap='viridis')
            axes[1, 1].set_title('Precision vs Recall')
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].colorbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        if not self.metrics_history:
            return "No training metrics available"
        
        df = pd.DataFrame(self.metrics_history)
        latest_metrics = df.iloc[-1]
        best_f1_idx = df['f1'].idxmax() if 'f1' in df.columns else 0
        best_metrics = df.iloc[best_f1_idx]
        
        report = []
        report.append("📊 PII TRAINING PERFORMANCE REPORT")
        report.append("=" * 50)
        
        report.append(f"\n🎯 FINAL PERFORMANCE (Epoch {latest_metrics['epoch']}):")
        report.append(f"  F1 Score: {latest_metrics.get('f1', 0):.4f}")
        report.append(f"  Precision: {latest_metrics.get('precision', 0):.4f}")
        report.append(f"  Recall: {latest_metrics.get('recall', 0):.4f}")
        
        report.append(f"\n🏆 BEST PERFORMANCE (Epoch {best_metrics['epoch']}):")
        report.append(f"  F1 Score: {best_metrics.get('f1', 0):.4f}")
        report.append(f"  Precision: {best_metrics.get('precision', 0):.4f}")
        report.append(f"  Recall: {best_metrics.get('recall', 0):.4f}")
        
        # Per-entity performance
        entity_metrics = [(col, latest_metrics[col]) for col in latest_metrics.index 
                         if col.endswith('_f1') and col != 'f1']
        
        if entity_metrics:
            report.append("\n📈 PER-ENTITY PERFORMANCE:")
            entity_metrics.sort(key=lambda x: x[1], reverse=True)
            for entity_col, f1_score in entity_metrics:
                entity_name = entity_col.replace('_f1', '')
                report.append(f"  {entity_name}: {f1_score:.4f}")
        
        return "\n".join(report)

def main():
    """Test the monitoring system"""
    monitor = PIITrainingMonitor()
    
    # Simulate some training metrics
    for epoch in range(1, 6):
        metrics = {
            'f1': 0.5 + epoch * 0.1 + np.random.normal(0, 0.02),
            'precision': 0.6 + epoch * 0.08 + np.random.normal(0, 0.02),
            'recall': 0.4 + epoch * 0.12 + np.random.normal(0, 0.02),
            'loss': 2.0 - epoch * 0.3 + np.random.normal(0, 0.1),
        }
        monitor.log_training_metrics(epoch, metrics)
    
    # Generate report
    report = monitor.generate_training_report()
    print(report)
    
    # Plot progress
    monitor.plot_training_progress()

if __name__ == "__main__":
    main()
