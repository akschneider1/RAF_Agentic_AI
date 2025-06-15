
import pandas as pd
import os

def inspect_wojood_dataset():
    """
    Load the Wojood dataset and inspect the first 50 rows,
    then print count of unique values in the entity_type column (tags).
    """
    folder_path = "Wojood/Wojood1_1_nested"
    
    print("WOJOOD DATASET INSPECTOR")
    print("="*40)
    
    # Load each split and inspect
    for filename in ['train.csv', 'val.csv', 'test.csv']:
        filepath = os.path.join(folder_path, filename)
        
        if os.path.exists(filepath):
            print(f"\n📁 Loading {filename}...")
            df = pd.read_csv(filepath)
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Inspect first 50 rows
            print(f"\n🔍 First 50 rows of {filename}:")
            print("-" * 60)
            print(df.head(50).to_string())
            
            # Count unique values in tags column (entity types)
            if 'tags' in df.columns:
                print(f"\n📈 Unique entity types in {filename}:")
                print("-" * 40)
                tag_counts = df['tags'].value_counts()
                print(f"Total unique tags: {len(tag_counts)}")
                print("\nTag distribution:")
                for tag, count in tag_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"{tag:15} : {count:6} ({percentage:5.2f}%)")
                
                # Separate analysis for named entities (non-O tags)
                entity_tags = df[df['tags'] != 'O']['tags'].value_counts()
                if len(entity_tags) > 0:
                    print(f"\n🏷️  Named entity types only (excluding 'O'):")
                    print("-" * 40)
                    for tag, count in entity_tags.items():
                        print(f"{tag:15} : {count:6}")
            
            print("\n" + "="*60)
        else:
            print(f"❌ File {filename} not found in {folder_path}")

if __name__ == "__main__":
    inspect_wojood_dataset()
