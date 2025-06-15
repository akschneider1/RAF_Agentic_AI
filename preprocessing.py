
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer
from dataclasses import dataclass
import torch

@dataclass
class ProcessedExample:
    """Represents a processed training example"""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    original_text: str
    word_ids: List[Optional[int]]

class ArabicTextNormalizer:
    """Normalizes Arabic text for NER processing"""
    
    def __init__(self):
        # Arabic diacritics to remove
        self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        
        # Character normalization mappings
        self.char_mappings = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variations
            'ة': 'ه',                        # Taa marbouta to haa
            'ي': 'ى',                        # Yaa variations
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Arabic text"""
        if not text:
            return text
        
        # Remove diacritics
        text = self.arabic_diacritics.sub('', text)
        
        # Normalize characters
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove extra punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text

class LabelAligner:
    """Handles alignment of word-level labels to subword tokens"""
    
    def __init__(self, label_to_id: Dict[str, int]):
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        
    def align_labels_with_tokens(
        self, 
        words: List[str], 
        labels: List[str], 
        tokenizer, 
        max_length: int = 512
    ) -> Tuple[List[int], List[int], List[int], List[Optional[int]]]:
        """
        Align word-level NER labels with subword tokens according to IOB2 scheme.
        
        Args:
            words: List of words
            labels: List of corresponding NER labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_ids, attention_mask, aligned_labels, word_ids)
        """
        # Tokenize each word individually to track alignment
        tokenized_inputs = tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        
        # Get word IDs for each token
        word_ids = tokenized_inputs.word_ids()
        
        # Initialize labels with -100 (ignore index)
        aligned_labels = [-100] * len(input_ids)
        
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels[token_idx] = -100
            elif word_idx != previous_word_idx:
                # First token of a word gets the original label
                if word_idx < len(labels):
                    label = labels[word_idx]
                    aligned_labels[token_idx] = self.label_to_id.get(label, self.label_to_id.get('O', 0))
            else:
                # Subsequent sub-tokens get -100 (ignored during loss calculation)
                aligned_labels[token_idx] = -100
            
            previous_word_idx = word_idx
        
        return input_ids, attention_mask, aligned_labels, word_ids
    
    def convert_iob_to_iob2(self, labels: List[str]) -> List[str]:
        """Convert IOB to IOB2 format ensuring proper B- prefixes"""
        if not labels:
            return labels
        
        converted = []
        prev_label = 'O'
        
        for label in labels:
            if label == 'O':
                converted.append('O')
                prev_label = 'O'
            elif label.startswith('I-'):
                entity_type = label[2:]
                prev_entity = prev_label[2:] if prev_label.startswith(('B-', 'I-')) else None
                
                if prev_entity == entity_type:
                    # Continue the entity
                    converted.append(label)
                else:
                    # Start new entity (convert I- to B-)
                    converted.append(f'B-{entity_type}')
                prev_label = converted[-1]
            elif label.startswith('B-'):
                converted.append(label)
                prev_label = label
            else:
                # Handle any other format
                converted.append(label)
                prev_label = label
        
        return converted

class NERPreprocessor:
    """Main preprocessing pipeline for NER data"""
    
    def __init__(self, model_name: str = 'aubmindlab/bert-base-arabertv2'):
        self.normalizer = ArabicTextNormalizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create label mappings
        self.label_to_id = self._create_label_mappings()
        self.aligner = LabelAligner(self.label_to_id)
        
    def _create_label_mappings(self) -> Dict[str, int]:
        """Create label to ID mappings for the 7 PII types"""
        labels = [
            'O',
            'B-PERSON', 'I-PERSON',
            'B-LOCATION', 'I-LOCATION', 
            'B-ORGANIZATION', 'I-ORGANIZATION',
            'B-PHONE', 'I-PHONE',
            'B-EMAIL', 'I-EMAIL',
            'B-ID_NUMBER', 'I-ID_NUMBER',
            'B-ADDRESS', 'I-ADDRESS'
        ]
        return {label: idx for idx, label in enumerate(labels)}
    
    def preprocess_sentence(
        self, 
        sentence_df: pd.DataFrame, 
        max_length: int = 512
    ) -> ProcessedExample:
        """
        Preprocess a single sentence from the dataset
        
        Args:
            sentence_df: DataFrame containing tokens and labels for one sentence
            max_length: Maximum sequence length
            
        Returns:
            ProcessedExample object
        """
        # Extract words and labels - handle different column names
        if 'token' in sentence_df.columns:
            words = sentence_df['token'].tolist()
        elif 'Token' in sentence_df.columns:
            words = sentence_df['Token'].tolist()
        else:
            raise ValueError("No token column found")
            
        if 'tag' in sentence_df.columns:
            labels = sentence_df['tag'].tolist()
        elif 'tags' in sentence_df.columns:
            labels = sentence_df['tags'].tolist()
        elif 'Tag' in sentence_df.columns:
            labels = sentence_df['Tag'].tolist()
        else:
            raise ValueError("No tag column found")
        
        # Normalize text
        normalized_words = [self.normalizer.normalize_text(word) for word in words]
        
        # Convert to IOB2 format
        iob2_labels = self.aligner.convert_iob_to_iob2(labels)
        
        # Reconstruct original text
        original_text = ' '.join(words)
        
        # Align labels with subword tokens
        input_ids, attention_mask, aligned_labels, word_ids = self.aligner.align_labels_with_tokens(
            normalized_words, iob2_labels, self.tokenizer, max_length
        )
        
        return ProcessedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=aligned_labels,
            original_text=original_text,
            word_ids=word_ids
        )
    
    def preprocess_dataset(
        self, 
        df: pd.DataFrame, 
        max_length: int = 512,
        sentence_id_col: str = 'sentence_id'
    ) -> List[ProcessedExample]:
        """
        Preprocess entire dataset
        
        Args:
            df: DataFrame with tokenized data
            max_length: Maximum sequence length
            sentence_id_col: Column name for sentence IDs
            
        Returns:
            List of ProcessedExample objects
        """
        processed_examples = []
        
        # Handle different sentence ID column names
        if sentence_id_col not in df.columns:
            if 'global_sentence_id' in df.columns:
                sentence_id_col = 'global_sentence_id'
            elif 'Sentence_ID' in df.columns:
                sentence_id_col = 'Sentence_ID'
            else:
                print(f"Warning: No sentence ID column found. Available columns: {df.columns.tolist()}")
                return processed_examples
        
        # Group by sentence
        sentence_groups = df.groupby(sentence_id_col)
        total_sentences = len(sentence_groups)
        processed_count = 0
        
        print(f"Processing {total_sentences} sentences...")
        
        for sentence_id, sentence_df in sentence_groups:
            try:
                # Sort by token position if available
                if 'token_position' in sentence_df.columns:
                    sentence_df = sentence_df.sort_values('token_position')
                elif 'token_id' in sentence_df.columns:
                    sentence_df = sentence_df.sort_values('token_id')
                elif 'Token_ID' in sentence_df.columns:
                    sentence_df = sentence_df.sort_values('Token_ID')
                
                processed_example = self.preprocess_sentence(sentence_df, max_length)
                processed_examples.append(processed_example)
                processed_count += 1
                
                # Progress update
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count}/{total_sentences} sentences...")
                
            except Exception as e:
                print(f"Error processing sentence {sentence_id}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_examples)} out of {total_sentences} sentences")
        return processed_examples
    
    def create_torch_dataset(self, processed_examples: List[ProcessedExample]) -> Dict[str, torch.Tensor]:
        """Convert processed examples to PyTorch tensors"""
        return {
            'input_ids': torch.tensor([ex.input_ids for ex in processed_examples], dtype=torch.long),
            'attention_mask': torch.tensor([ex.attention_mask for ex in processed_examples], dtype=torch.long),
            'labels': torch.tensor([ex.labels for ex in processed_examples], dtype=torch.long)
        }
    
    def analyze_preprocessing_stats(self, processed_examples: List[ProcessedExample]):
        """Analyze preprocessing statistics"""
        print("PREPROCESSING STATISTICS")
        print("=" * 50)
        
        # Sequence length statistics
        seq_lengths = [len([l for l in ex.labels if l != -100]) for ex in processed_examples]
        print(f"Processed examples: {len(processed_examples)}")
        print(f"Average sequence length: {np.mean(seq_lengths):.2f}")
        print(f"Max sequence length: {max(seq_lengths)}")
        print(f"Min sequence length: {min(seq_lengths)}")
        
        # Label distribution
        all_labels = []
        for ex in processed_examples:
            all_labels.extend([l for l in ex.labels if l != -100])
        
        label_counts = {}
        for label_id in all_labels:
            label_name = self.aligner.id_to_label.get(label_id, f"UNK_{label_id}")
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")
        
        # Entity statistics
        entity_labels = {k: v for k, v in label_counts.items() if k != 'O'}
        total_entities = sum(entity_labels.values())
        total_tokens = len(all_labels)
        
        print(f"\nEntity ratio: {total_entities}/{total_tokens} ({total_entities/total_tokens*100:.2f}%)")

def create_preprocessing_pipeline():
    """Create and test the preprocessing pipeline"""
    print("CREATING NER PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = NERPreprocessor()
    
    # Load augmented training data
    if os.path.exists('train_augmented.csv'):
        print("Loading augmented training data...")
        df = pd.read_csv('train_augmented.csv')
    else:
        print("Loading original Wojood training data...")
        df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        # Add sentence_id column if not present
        if 'sentence_id' not in df.columns:
            df['sentence_id'] = df['global_sentence_id']
    
    print(f"Loaded {len(df)} tokens")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:")
    print(df.head())
    
    # Filter to only include rows with valid labels
    print("Filtering valid data...")
    initial_len = len(df)
    df = df.dropna(subset=['tags' if 'tags' in df.columns else 'token'])
    print(f"Filtered from {initial_len} to {len(df)} rows")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    processed_examples = preprocessor.preprocess_dataset(df)  # Process all data
    
    # Analyze statistics
    preprocessor.analyze_preprocessing_stats(processed_examples)
    
    # Create torch dataset
    torch_dataset = preprocessor.create_torch_dataset(processed_examples)
    print(f"\nTorch dataset shapes:")
    for key, tensor in torch_dataset.items():
        print(f"  {key}: {tensor.shape}")
    
    # Save preprocessor settings
    import pickle
    with open('preprocessor_config.pkl', 'wb') as f:
        pickle.dump({
            'label_to_id': preprocessor.label_to_id,
            'tokenizer_name': 'aubmindlab/bert-base-arabertv2'
        }, f)
    
    print("Preprocessor configuration saved to preprocessor_config.pkl")
    
    return preprocessor, processed_examples

def test_label_alignment():
    """Test the label alignment function with examples"""
    print("\nTESTING LABEL ALIGNMENT")
    print("-" * 30)
    
    # Test case 1: Simple example
    words = ['أحمد', 'يعمل', 'في', 'شركة', 'مايكروسوفت']
    labels = ['B-PERSON', 'O', 'O', 'O', 'B-ORGANIZATION']
    
    preprocessor = NERPreprocessor()
    
    input_ids, attention_mask, aligned_labels, word_ids = preprocessor.aligner.align_labels_with_tokens(
        words, labels, preprocessor.tokenizer, max_length=20
    )
    
    print("Test Case 1:")
    print(f"Words: {words}")
    print(f"Labels: {labels}")
    print(f"Tokens: {preprocessor.tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"Word IDs: {word_ids}")
    print(f"Aligned labels: {[preprocessor.aligner.id_to_label.get(l, l) for l in aligned_labels]}")
    
    # Test case 2: Arabic text with subwords
    words = ['عبدالرحمن', 'اتصل', 'على', '+966501234567']
    labels = ['B-PERSON', 'O', 'O', 'B-PHONE']
    
    input_ids, attention_mask, aligned_labels, word_ids = preprocessor.aligner.align_labels_with_tokens(
        words, labels, preprocessor.tokenizer, max_length=20
    )
    
    print("\nTest Case 2:")
    print(f"Words: {words}")
    print(f"Labels: {labels}")
    print(f"Tokens: {preprocessor.tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"Word IDs: {word_ids}")
    print(f"Aligned labels: {[preprocessor.aligner.id_to_label.get(l, l) for l in aligned_labels]}")

if __name__ == "__main__":
    import os
    test_label_alignment()
    preprocessor, examples = create_preprocessing_pipeline()
