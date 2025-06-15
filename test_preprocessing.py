
import pandas as pd
from preprocessing import NERPreprocessor, test_label_alignment, create_preprocessing_pipeline

def test_full_pipeline():
    """Test the complete preprocessing pipeline"""
    print("TESTING FULL PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Test with sample data
    sample_data = {
        'sentence_id': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        'token_id': [0, 1, 2, 3, 4, 0, 1, 2, 3],
        'token': ['أحمد', 'يسكن', 'في', 'الرياض', 'ويعمل', 'هاتف', 'محمد', '0501234567', '.'],
        'tag': ['B-PERSON', 'O', 'O', 'B-LOCATION', 'O', 'O', 'B-PERSON', 'B-PHONE', 'O']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data:")
    print(df)
    
    # Initialize preprocessor
    preprocessor = NERPreprocessor()
    
    # Preprocess the sample
    processed_examples = preprocessor.preprocess_dataset(df, max_length=32)
    
    print(f"\nProcessed {len(processed_examples)} sentences")
    
    # Show results for each sentence
    for i, example in enumerate(processed_examples):
        print(f"\nSentence {i+1}:")
        print(f"Original: {example.original_text}")
        
        tokens = preprocessor.tokenizer.convert_ids_to_tokens(example.input_ids)
        labels = [preprocessor.aligner.id_to_label.get(l, str(l)) for l in example.labels]
        
        print("Tokens and labels:")
        for token, label, word_id in zip(tokens, labels, example.word_ids):
            if token != '[PAD]':
                print(f"  {token:15} -> {label:12} (word_id: {word_id})")

def test_arabic_normalization():
    """Test Arabic text normalization"""
    print("\nTESTING ARABIC NORMALIZATION")
    print("-" * 30)
    
    from preprocessing import ArabicTextNormalizer
    normalizer = ArabicTextNormalizer()
    
    test_texts = [
        'أحمد بن عبدالله',  # Alef variations
        'مُحَمَّد',           # With diacritics
        'الشركة والمؤسسة',   # Multiple spaces
        'هذه   مشكلة....',   # Extra punctuation
        'عائشة المحترمة',    # Taa marbouta
    ]
    
    for text in test_texts:
        normalized = normalizer.normalize_text(text)
        print(f"'{text}' -> '{normalized}'")

if __name__ == "__main__":
    test_arabic_normalization()
    test_full_pipeline()
    test_label_alignment()
    create_preprocessing_pipeline()
