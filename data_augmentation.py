
import pandas as pd
import json
from synthetic_generator import SyntheticPIIGenerator
from schema_mapper import WojoodToPIIMapper
from typing import List, Dict
import random

class DataAugmentation:
    """Creates augmented training data combining Wojood and synthetic data"""
    
    def __init__(self):
        self.generator = SyntheticPIIGenerator()
        self.mapper = WojoodToPIIMapper()
    
    def prepare_wojood_data(self) -> pd.DataFrame:
        """Prepare Wojood training data in our format"""
        print("Loading Wojood training data...")
        
        # Load original training data
        wojood_df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        
        # Extract PII entities from Wojood
        entities = self.mapper.extract_pii_entities(wojood_df)
        
        # Create sentence-level records
        wojood_records = []
        
        for sentence_id in wojood_df['global_sentence_id'].unique():
            sentence_df = wojood_df[wojood_df['global_sentence_id'] == sentence_id].sort_values('token_id')
            
            # Reconstruct sentence text
            sentence_text = ' '.join(sentence_df['token'].tolist())
            
            # Find PII entities in this sentence
            sentence_pii = []
            current_entity = None
            current_tokens = []
            current_start = 0
            
            for idx, row in sentence_df.iterrows():
                tag = row['tags']
                token = row['token']
                pii_type = self.mapper.map_entity(tag)
                
                if tag.startswith('B-') and pii_type != 'OTHER':
                    # Start of new entity - save previous if exists
                    if current_entity and current_tokens:
                        entity_text = ' '.join(current_tokens)
                        sentence_pii.append({
                            'text': entity_text,
                            'type': current_entity,
                            'start': current_start,
                            'end': current_start + len(entity_text)
                        })
                    
                    # Start new entity
                    current_entity = pii_type
                    current_tokens = [token]
                    current_start = sentence_text.find(token, current_start)
                    
                elif tag.startswith('I-') and current_entity and pii_type == current_entity:
                    # Continue current entity
                    current_tokens.append(token)
                    
                else:
                    # End entity if exists
                    if current_entity and current_tokens:
                        entity_text = ' '.join(current_tokens)
                        sentence_pii.append({
                            'text': entity_text,
                            'type': current_entity,
                            'start': current_start,
                            'end': current_start + len(entity_text)
                        })
                    current_entity = None
                    current_tokens = []
            
            # Add final entity if exists
            if current_entity and current_tokens:
                entity_text = ' '.join(current_tokens)
                sentence_pii.append({
                    'text': entity_text,
                    'type': current_entity,
                    'start': current_start,
                    'end': current_start + len(entity_text)
                })
            
            wojood_records.append({
                'sentence_id': f"wojood_{sentence_id}",
                'text': sentence_text,
                'source': 'wojood',
                'pii_entities': sentence_pii,
                'subcorpus': sentence_df['Sub_corpus'].iloc[0] if 'Sub_corpus' in sentence_df.columns else 'unknown'
            })
        
        return pd.DataFrame(wojood_records)
    
    def generate_synthetic_data(self, num_sentences: int = 20000) -> pd.DataFrame:
        """Generate synthetic training data"""
        print(f"Generating {num_sentences} synthetic sentences...")
        
        synthetic_data = self.generator.generate_dataset(num_sentences)
        
        # Convert to our format
        synthetic_records = []
        
        for data in synthetic_data:
            # Convert detected PII to our format
            pii_entities = []
            for pii in data['detected_pii']:
                pii_entities.append({
                    'text': pii['text'],
                    'type': pii['type'],
                    'start': pii['start'],
                    'end': pii['end']
                })
            
            synthetic_records.append({
                'sentence_id': f"synthetic_{data['sentence_id']}",
                'text': data['text'],
                'source': 'synthetic',
                'pii_entities': pii_entities,
                'template_category': data['template_category']
            })
        
        return pd.DataFrame(synthetic_records)
    
    def create_augmented_dataset(self, synthetic_ratio: float = 0.7) -> pd.DataFrame:
        """Create combined augmented dataset"""
        print("Creating augmented training dataset...")
        
        # Get Wojood data
        wojood_df = self.prepare_wojood_data()
        print(f"Wojood sentences: {len(wojood_df)}")
        
        # Calculate how many synthetic sentences we need
        wojood_count = len(wojood_df)
        if synthetic_ratio > 0:
            synthetic_count = int(wojood_count * synthetic_ratio / (1 - synthetic_ratio))
        else:
            synthetic_count = 0
        
        # Generate synthetic data
        if synthetic_count > 0:
            synthetic_df = self.generate_synthetic_data(synthetic_count)
            print(f"Synthetic sentences: {len(synthetic_df)}")
            
            # Combine datasets
            augmented_df = pd.concat([wojood_df, synthetic_df], ignore_index=True)
        else:
            augmented_df = wojood_df
        
        # Shuffle the dataset
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Total augmented sentences: {len(augmented_df)}")
        
        return augmented_df
    
    def convert_to_token_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert sentence-level data to token-level BIO format"""
        print("Converting to token-level BIO format...")
        
        token_records = []
        
        for idx, row in df.iterrows():
            sentence_text = row['text']
            pii_entities = row['pii_entities']
            
            # Tokenize (simple whitespace tokenization)
            tokens = sentence_text.split()
            token_positions = []
            current_pos = 0
            
            for token in tokens:
                start_pos = sentence_text.find(token, current_pos)
                end_pos = start_pos + len(token)
                token_positions.append((token, start_pos, end_pos))
                current_pos = end_pos
            
            # Create BIO tags
            bio_tags = ['O'] * len(tokens)
            
            for entity in pii_entities:
                entity_start = entity['start']
                entity_end = entity['end']
                entity_type = entity['type']
                
                # Find tokens that overlap with this entity
                first_token_idx = None
                for i, (token, token_start, token_end) in enumerate(token_positions):
                    if token_start < entity_end and token_end > entity_start:
                        if first_token_idx is None:
                            first_token_idx = i
                            bio_tags[i] = f"B-{entity_type}"
                        else:
                            bio_tags[i] = f"I-{entity_type}"
            
            # Create token records
            for i, (token, token_start, token_end) in enumerate(token_positions):
                token_records.append({
                    'sentence_id': row['sentence_id'],
                    'token_id': i,
                    'token': token,
                    'tag': bio_tags[i],
                    'source': row['source'],
                    'start_pos': token_start,
                    'end_pos': token_end
                })
        
        return pd.DataFrame(token_records)
    
    def analyze_augmented_data(self, df: pd.DataFrame):
        """Analyze the augmented dataset"""
        print("\nAUGMENTED DATASET ANALYSIS")
        print("=" * 50)
        
        # Source distribution
        source_dist = df['source'].value_counts()
        print(f"Source distribution:")
        for source, count in source_dist.items():
            print(f"  {source}: {count} ({count/len(df)*100:.1f}%)")
        
        # PII tag distribution
        tag_dist = df['tag'].value_counts()
        print(f"\nTag distribution (top 20):")
        for tag, count in tag_dist.head(20).items():
            print(f"  {tag}: {count}")
        
        # Entity vs non-entity ratio
        entity_count = (df['tag'] != 'O').sum()
        total_count = len(df)
        print(f"\nEntity vs Non-entity ratio:")
        print(f"  Entities: {entity_count} ({entity_count/total_count*100:.1f}%)")
        print(f"  Non-entities (O): {total_count - entity_count} ({(total_count - entity_count)/total_count*100:.1f}%)")
        
        # PII type distribution
        entity_df = df[df['tag'] != 'O']
        if len(entity_df) > 0:
            pii_types = entity_df['tag'].str.replace(r'^[BI]-', '', regex=True).value_counts()
            print(f"\nPII type distribution:")
            for pii_type, count in pii_types.items():
                print(f"  {pii_type}: {count}")

def create_train_augmented():
    """Main function to create train_augmented.csv"""
    print("CREATING AUGMENTED TRAINING DATASET")
    print("=" * 50)
    
    augmenter = DataAugmentation()
    
    # Create augmented dataset (70% synthetic, 30% Wojood)
    augmented_df = augmenter.create_augmented_dataset(synthetic_ratio=0.7)
    
    # Convert to token-level format
    token_df = augmenter.convert_to_token_format(augmented_df)
    
    # Analyze the results
    augmenter.analyze_augmented_data(token_df)
    
    # Save to CSV
    output_file = 'train_augmented.csv'
    token_df.to_csv(output_file, index=False)
    print(f"\nAugmented dataset saved to {output_file}")
    print(f"Total tokens: {len(token_df)}")
    
    # Save sentence-level data for reference
    sentence_file = 'train_augmented_sentences.csv'
    augmented_df.to_csv(sentence_file, index=False)
    print(f"Sentence-level data saved to {sentence_file}")
    
    return token_df

if __name__ == "__main__":
    create_train_augmented()
