
#!/usr/bin/env python3
"""
Complete pipeline to create augmented training data for PII detection
"""

from schema_mapper import analyze_wojood_pii_mapping
from synthetic_generator import test_synthetic_generator
from data_augmentation import create_train_augmented
import sys

def main():
    """Run the complete augmentation pipeline"""
    print("🚀 PII DATA AUGMENTATION PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Analyze Wojood mapping
        print("\n📊 STEP 1: Analyzing Wojood to PII mapping...")
        entities = analyze_wojood_pii_mapping()
        
        # Step 2: Test synthetic generator
        print("\n🔧 STEP 2: Testing synthetic data generator...")
        test_synthetic_generator()
        
        # Step 3: Create augmented dataset
        print("\n📈 STEP 3: Creating augmented training dataset...")
        augmented_data = create_train_augmented()
        
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("  📁 train_augmented.csv - Token-level training data")
        print("  📁 train_augmented_sentences.csv - Sentence-level data")
        print("  📁 entity_distribution.png - Visualization")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
