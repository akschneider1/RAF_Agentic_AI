
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_wojood_dataset(folder_path="Wojood/Wojood1_1_nested"):
    """Load all CSV files from the Wojood dataset folder"""
    datasets = {}
    
    for filename in ['train.csv', 'val.csv', 'test.csv']:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)
            datasets[filename.replace('.csv', '')] = df
            print(f"Loaded {len(df)} rows from {filename}")
        else:
            print(f"File {filename} not found in {folder_path}")
    
    return datasets

def analyze_dataset_structure(datasets):
    """Analyze the structure of the dataset"""
    print("\n" + "="*50)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*50)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Column types:\n{df.dtypes}")
        
        # Show first few rows
        print(f"\nFirst 5 rows of {split_name}:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values in {split_name}:")
        print(df.isnull().sum())

def analyze_entity_distribution(datasets):
    """Analyze entity distribution across the dataset"""
    print("\n" + "="*50)
    print("ENTITY DISTRIBUTION ANALYSIS")
    print("="*50)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset Entity Analysis:")
        
        # Count unique values in the tags column
        if 'tags' in df.columns:
            tag_counts = df['tags'].value_counts()
            print(f"Total unique tags: {len(tag_counts)}")
            print(f"Tag distribution:")
            print(tag_counts.head(20))  # Show top 20 tags
            
            # Separate entity tags from O tags
            entity_tags = df[df['tags'] != 'O']['tags'].value_counts()
            print(f"\nNon-O entity tags: {len(entity_tags)}")
            print(entity_tags.head(15))
            
            # Calculate entity vs non-entity ratio
            o_count = (df['tags'] == 'O').sum()
            entity_count = len(df) - o_count
            print(f"\nEntity vs Non-entity ratio:")
            print(f"O (non-entity): {o_count} ({o_count/len(df)*100:.2f}%)")
            print(f"Entities: {entity_count} ({entity_count/len(df)*100:.2f}%)")

def analyze_text_formats(datasets):
    """Analyze text formats and linguistic patterns"""
    print("\n" + "="*50)
    print("TEXT FORMAT ANALYSIS")
    print("="*50)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset Text Analysis:")
        
        if 'token' in df.columns:
            # Token length analysis
            df['token_length'] = df['token'].str.len()
            print(f"Token length statistics:")
            print(df['token_length'].describe())
            
            # Character analysis
            all_text = ' '.join(df['token'].astype(str))
            print(f"\nTotal characters: {len(all_text)}")
            print(f"Unique characters: {len(set(all_text))}")
            
            # Language detection (simple heuristic)
            arabic_chars = sum(1 for char in all_text if '\u0600' <= char <= '\u06FF')
            english_chars = sum(1 for char in all_text if char.isascii() and char.isalpha())
            
            print(f"Arabic characters: {arabic_chars}")
            print(f"English characters: {english_chars}")
            
            # Sample tokens
            print(f"\nSample tokens:")
            print(df['token'].head(20).tolist())

def analyze_subcorpora(datasets):
    """Analyze subcorpora distribution"""
    print("\n" + "="*50)
    print("SUBCORPORA ANALYSIS")
    print("="*50)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset Subcorpora:")
        
        if 'Sub_corpus' in df.columns:
            subcorpus_counts = df['Sub_corpus'].value_counts()
            print(f"Subcorpora distribution:")
            print(subcorpus_counts)
            
            # Analyze entity distribution per subcorpus
            if 'tags' in df.columns:
                print(f"\nEntity distribution per subcorpus:")
                for subcorpus in subcorpus_counts.index:
                    subset = df[df['Sub_corpus'] == subcorpus]
                    entity_count = (subset['tags'] != 'O').sum()
                    total_count = len(subset)
                    print(f"{subcorpus}: {entity_count}/{total_count} entities ({entity_count/total_count*100:.2f}%)")

def analyze_sentences(datasets):
    """Analyze sentence-level statistics"""
    print("\n" + "="*50)
    print("SENTENCE-LEVEL ANALYSIS")
    print("="*50)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset Sentence Analysis:")
        
        if 'global_sentence_id' in df.columns:
            sentence_lengths = df.groupby('global_sentence_id').size()
            print(f"Sentence length statistics (tokens per sentence):")
            print(sentence_lengths.describe())
            
            print(f"Total sentences: {len(sentence_lengths)}")
            print(f"Average tokens per sentence: {sentence_lengths.mean():.2f}")
            print(f"Longest sentence: {sentence_lengths.max()} tokens")
            print(f"Shortest sentence: {sentence_lengths.min()} tokens")

def create_entity_visualization(datasets):
    """Create visualizations for entity distribution"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    try:
        # Combine all datasets for overall analysis
        all_data = []
        for split_name, df in datasets.items():
            df_copy = df.copy()
            df_copy['split'] = split_name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create entity type visualization
        if 'tags' in combined_df.columns:
            # Get top entity types (excluding 'O')
            entity_tags = combined_df[combined_df['tags'] != 'O']['tags'].value_counts().head(15)
            
            plt.figure(figsize=(12, 8))
            entity_tags.plot(kind='bar')
            plt.title('Top 15 Entity Types Distribution')
            plt.xlabel('Entity Tags')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('entity_distribution.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("Entity distribution chart saved as 'entity_distribution.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main function to run all analyses"""
    print("WOJOOD NER DATASET ANALYSIS")
    print("="*50)
    
    # Load datasets
    datasets = load_wojood_dataset()
    
    if not datasets:
        print("No datasets found. Please check the file paths.")
        return
    
    # Run all analyses
    analyze_dataset_structure(datasets)
    analyze_entity_distribution(datasets)
    analyze_text_formats(datasets)
    analyze_subcorpora(datasets)
    analyze_sentences(datasets)
    
    # Create visualizations
    create_entity_visualization(datasets)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
