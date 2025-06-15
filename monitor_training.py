
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def monitor_training_progress(checkpoint_dir="./model_checkpoints"):
    """Monitor and visualize training progress"""
    print("📊 MONITORING TRAINING PROGRESS")
    print("=" * 50)
    
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print("No training state found. Training hasn't started yet.")
        return
    
    # Load training state
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    # Extract training history
    log_history = trainer_state.get('log_history', [])
    
    if not log_history:
        print("No training logs found yet.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(log_history)
    
    # Separate training and validation logs
    train_logs = df[df['train_loss'].notna()].copy()
    eval_logs = df[df['eval_loss'].notna()].copy()
    
    print(f"Training steps completed: {len(train_logs)}")
    print(f"Evaluation steps completed: {len(eval_logs)}")
    
    if len(train_logs) > 0:
        print(f"Current training loss: {train_logs['train_loss'].iloc[-1]:.4f}")
    
    if len(eval_logs) > 0:
        latest_eval = eval_logs.iloc[-1]
        print(f"Current validation loss: {latest_eval['eval_loss']:.4f}")
        if 'eval_f1' in latest_eval:
            print(f"Current F1 score: {latest_eval['eval_f1']:.4f}")
        if 'eval_precision' in latest_eval:
            print(f"Current Precision: {latest_eval['eval_precision']:.4f}")
        if 'eval_recall' in latest_eval:
            print(f"Current Recall: {latest_eval['eval_recall']:.4f}")
    
    # Create visualizations
    if len(train_logs) > 1:
        plt.figure(figsize=(15, 5))
        
        # Training loss plot
        plt.subplot(1, 3, 1)
        plt.plot(train_logs['step'], train_logs['train_loss'], 'b-', label='Training Loss')
        if len(eval_logs) > 0:
            plt.plot(eval_logs['step'], eval_logs['eval_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # F1 Score plot
        if len(eval_logs) > 0 and 'eval_f1' in eval_logs.columns:
            plt.subplot(1, 3, 2)
            plt.plot(eval_logs['step'], eval_logs['eval_f1'], 'g-', label='F1 Score')
            plt.xlabel('Steps')
            plt.ylabel('F1 Score')
            plt.title('F1 Score Progress')
            plt.legend()
            plt.grid(True)
        
        # Precision and Recall plot
        if len(eval_logs) > 0 and 'eval_precision' in eval_logs.columns:
            plt.subplot(1, 3, 3)
            plt.plot(eval_logs['step'], eval_logs['eval_precision'], 'orange', label='Precision')
            plt.plot(eval_logs['step'], eval_logs['eval_recall'], 'purple', label='Recall')
            plt.xlabel('Steps')
            plt.ylabel('Score')
            plt.title('Precision & Recall')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training progress chart saved as 'training_progress.png'")

if __name__ == "__main__":
    monitor_training_progress()
