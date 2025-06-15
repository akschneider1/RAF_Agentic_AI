
import pandas as pd
import os
from collections import defaultdict
import re

def analyze_entity_patterns():
    """Analyze entity patterns and BIO tagging scheme"""
    folder_path = "Wojood/Wojood1_1_nested"
    
    print("ENTITY PATTERN ANALYZER")
    print("="*50)
    
    all_datasets = {}
    
    # Load all datasets
    for filename in ['train.csv', 'val.csv', 'test.csv']:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            all_datasets[filename] = pd.read_csv(filepath)
    
    for filename, df in all_datasets.items():
        print(f"\n📊 Analyzing {filename}")
        print("-" * 30)
        
        # Extract entity types from BIO tags
        entity_types = set()
        bio_pattern = re.compile(r'^[BIO]-(.+)$')
        
        for tag in df['tags'].unique():
            if tag != 'O':
                match = bio_pattern.match(tag)
                if match:
                    entity_types.add(match.group(1))
                else:
                    entity_types.add(tag)  # In case there are non-BIO tags
        
        print(f"🏷️  Entity Types Found: {len(entity_types)}")
        for entity_type in sorted(entity_types):
            print(f"   - {entity_type}")
        
        # Analyze BIO distribution
        bio_stats = defaultdict(int)
        for tag in df['tags']:
            if tag.startswith('B-'):
                bio_stats['B'] += 1
            elif tag.startswith('I-'):
                bio_stats['I'] += 1
            elif tag == 'O':
                bio_stats['O'] += 1
            else:
                bio_stats['Other'] += 1
        
        print(f"\n📈 BIO Tag Distribution:")
        total_tokens = len(df)
        for bio_type, count in bio_stats.items():
            percentage = (count / total_tokens) * 100
            print(f"   {bio_type}: {count:6} ({percentage:5.2f}%)")
        
        # Find entity spans
        entity_spans = []
        current_entity = None
        current_tokens = []
        
        for idx, row in df.iterrows():
            tag = row['tags']
            token = row['token']
            
            if tag.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entity_spans.append((current_entity, ' '.join(current_tokens)))
                current_entity = tag[2:]  # Remove 'B-'
                current_tokens = [token]
            elif tag.startswith('I-') and current_entity:
                # Continue current entity
                current_tokens.append(token)
            else:
                # End of entity or non-entity token
                if current_entity:
                    entity_spans.append((current_entity, ' '.join(current_tokens)))
                current_entity = None
                current_tokens = []
        
        # Add last entity if exists
        if current_entity:
            entity_spans.append((current_entity, ' '.join(current_tokens)))
        
        print(f"\n🎯 Entity Spans Found: {len(entity_spans)}")
        
        # Show examples of each entity type
        entity_examples = defaultdict(list)
        for entity_type, entity_text in entity_spans:
            if len(entity_examples[entity_type]) < 5:  # Limit examples
                entity_examples[entity_type].append(entity_text)
        
        print(f"\n📝 Entity Examples:")
        for entity_type in sorted(entity_examples.keys()):
            print(f"   {entity_type}:")
            for example in entity_examples[entity_type]:
                print(f"      '{example}'")

def analyze_corpus_distribution():
    """Analyze distribution across different subcorpora"""
    folder_path = "Wojood/Wojood1_1_nested"
    
    print("\n" + "="*50)
    print("CORPUS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    for filename in ['train.csv', 'val.csv', 'test.csv']:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            print(f"\n📊 {filename} Corpus Analysis:")
            print("-" * 30)
            
            if 'Sub_corpus' in df.columns:
                corpus_stats = df.groupby('Sub_corpus').agg({
                    'token': 'count',
                    'tags': lambda x: (x != 'O').sum()
                }).rename(columns={'token': 'total_tokens', 'tags': 'entity_tokens'})
                
                corpus_stats['entity_ratio'] = (corpus_stats['entity_tokens'] / corpus_stats['total_tokens'] * 100).round(2)
                
                print(corpus_stats)

if __name__ == "__main__":
    analyze_entity_patterns()
    analyze_corpus_distribution()
