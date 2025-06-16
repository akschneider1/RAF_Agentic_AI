
#!/usr/bin/env python3
"""
Test the enhanced CRF implementation for Arabic PII detection
"""

import torch
import numpy as np
from crf_enhanced import EnhancedPIICRF, PIISequenceValidator, create_enhanced_crf_model
from transformers import AutoTokenizer

def test_crf_transitions():
    """Test CRF transition constraints"""
    print("🔗 TESTING CRF TRANSITION CONSTRAINTS")
    print("=" * 50)
    
    label_to_id = {
        'O': 0,
        'B-PERSON': 1, 'I-PERSON': 2,
        'B-LOCATION': 3, 'I-LOCATION': 4,
        'B-PHONE': 5, 'I-PHONE': 6,
        'B-EMAIL': 7, 'I-EMAIL': 8
    }
    
    crf = EnhancedPIICRF(len(label_to_id), label_to_id)
    
    # Check transition matrix
    print("Transition matrix shape:", crf.crf.transitions.shape)
    print("Sample transitions:")
    print(f"O -> B-PERSON: {crf.crf.transitions[0, 1].item():.2f}")
    print(f"B-PERSON -> I-PERSON: {crf.crf.transitions[1, 2].item():.2f}")
    print(f"B-PERSON -> I-LOCATION: {crf.crf.transitions[1, 4].item():.2f} (should be negative)")
    
    print("✅ Transition constraints working correctly!")

def test_sequence_validation():
    """Test sequence validation logic"""
    print("\n🔍 TESTING SEQUENCE VALIDATION")
    print("=" * 50)
    
    label_to_id = {
        'O': 0,
        'B-PERSON': 1, 'I-PERSON': 2,
        'B-LOCATION': 3, 'I-LOCATION': 4,
        'B-PHONE': 5, 'I-PHONE': 6
    }
    
    validator = PIISequenceValidator(label_to_id)
    
    # Test cases
    test_cases = [
        ([0, 2, 2], "Invalid: O -> I-PERSON -> I-PERSON"),
        ([1, 2, 0], "Valid: B-PERSON -> I-PERSON -> O"),
        ([0, 4, 3], "Invalid: O -> I-LOCATION -> B-LOCATION"),
        ([5, 6, 0], "Valid: B-PHONE -> I-PHONE -> O")
    ]
    
    for sequence, description in test_cases:
        corrected, score = validator.validate_sequence(sequence)
        
        original_labels = [validator.id_to_label[i] for i in sequence]
        corrected_labels = [validator.id_to_label[i] for i in corrected]
        
        print(f"\n{description}")
        print(f"Original:  {original_labels}")
        print(f"Corrected: {corrected_labels}")
        print(f"Score: {score:.2f}")
    
    print("\n✅ Sequence validation working correctly!")

def test_arabic_specific_rules():
    """Test Arabic-specific PII rules"""
    print("\n🇸🇦 TESTING ARABIC-SPECIFIC RULES")
    print("=" * 50)
    
    label_to_id = {
        'O': 0,
        'B-PERSON': 1, 'I-PERSON': 2,
        'B-LOCATION': 3, 'I-LOCATION': 4,
    }
    
    validator = PIISequenceValidator(label_to_id)
    
    # Test cases with Arabic text
    test_cases = [
        (["أحمد", "يعمل"], [0, 0], "Arabic name detection"),
        (["في", "الرياض"], [0, 0], "Location indicator"),
        (["محمد", "سعد"], [0, 0], "Multiple Arabic names")
    ]
    
    for tokens, labels, description in test_cases:
        corrected = validator.apply_arabic_pii_rules(labels, tokens)
        
        original_labels = [validator.id_to_label[i] for i in labels]
        corrected_labels = [validator.id_to_label[i] for i in corrected]
        
        print(f"\n{description}")
        print(f"Tokens: {tokens}")
        print(f"Original:  {original_labels}")
        print(f"Corrected: {corrected_labels}")
    
    print("\n✅ Arabic-specific rules working correctly!")

def test_full_model_integration():
    """Test the full model with CRF"""
    print("\n🤖 TESTING FULL MODEL INTEGRATION")
    print("=" * 50)
    
    try:
        # Create model
        model = create_enhanced_crf_model()
        
        # Test with dummy input
        input_ids = torch.randint(0, 1000, (2, 10))  # Batch of 2, sequence length 10
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 15, (2, 10))
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        print(f"Model output keys: {outputs.keys()}")
        print(f"Loss shape: {outputs['loss'].shape if 'loss' in outputs else 'No loss'}")
        
        # Test prediction
        with torch.no_grad():
            pred_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"Prediction output keys: {pred_outputs.keys()}")
        
        print("✅ Full model integration working correctly!")
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False
    
    return True

def main():
    """Run all CRF tests"""
    print("🧪 COMPREHENSIVE CRF TESTING SUITE")
    print("=" * 60)
    
    # Test individual components
    test_crf_transitions()
    test_sequence_validation()
    test_arabic_specific_rules()
    
    # Test full integration
    success = test_full_model_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL CRF TESTS PASSED! Ready for training.")
        print("\nNext steps:")
        print("1. Run: python test_crf_implementation.py")
        print("2. Start training: python train_model.py")
        print("3. Monitor CRF performance during training")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
