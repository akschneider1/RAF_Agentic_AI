
import pandas as pd
import re
from typing import Dict, List, Tuple
from collections import defaultdict

class WojoodToPIIMapper:
    """Maps Wojood NER entities to our 7 PII categories"""
    
    def __init__(self):
        self.mapping = self._create_mapping()
    
    def _create_mapping(self) -> Dict[str, str]:
        """Create mapping from Wojood entity types to our PII categories"""
        return {
            # PERSON mapping
            'PERS': 'PERSON',
            'B-PERS': 'PERSON',
            'I-PERS': 'PERSON',
            
            # LOCATION mapping (includes addresses)
            'GPE': 'LOCATION',
            'B-GPE': 'LOCATION', 
            'I-GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'B-LOC': 'LOCATION',
            'I-LOC': 'LOCATION',
            
            # ORGANIZATION mapping
            'ORG': 'ORGANIZATION',
            'B-ORG': 'ORGANIZATION',
            'I-ORG': 'ORGANIZATION',
            
            # PHONE numbers (from context analysis)
            'PHONE': 'PHONE',
            'B-PHONE': 'PHONE',
            'I-PHONE': 'PHONE',
            
            # EMAIL addresses  
            'EMAIL': 'EMAIL',
            'B-EMAIL': 'EMAIL',
            'I-EMAIL': 'EMAIL',
            
            # ID_NUMBER mapping (from various ID types)
            'ID': 'ID_NUMBER',
            'B-ID': 'ID_NUMBER',
            'I-ID': 'ID_NUMBER',
            'NATIONAL_ID': 'ID_NUMBER',
            'PASSPORT': 'ID_NUMBER',
            'LICENSE': 'ID_NUMBER',
            
            # ADDRESS mapping (specific location references)
            'ADDRESS': 'ADDRESS',
            'B-ADDRESS': 'ADDRESS',
            'I-ADDRESS': 'ADDRESS',
            'STREET': 'ADDRESS',
            'BUILDING': 'ADDRESS'
        }
    
    def map_entity(self, wojood_tag: str) -> str:
        """Map a single Wojood tag to PII category"""
        # Remove BIO prefix for mapping
        base_tag = wojood_tag
        if wojood_tag.startswith(('B-', 'I-')):
            base_tag = wojood_tag[2:]
        
        # Direct mapping
        if wojood_tag in self.mapping:
            return self.mapping[wojood_tag]
        if base_tag in self.mapping:
            return self.mapping[base_tag]
        
        # Fuzzy matching for complex tags
        if any(person_word in base_tag.upper() for person_word in ['PERS', 'PERSON', 'NAME']):
            return 'PERSON'
        elif any(loc_word in base_tag.upper() for loc_word in ['GPE', 'LOC', 'PLACE', 'CITY', 'COUNTRY']):
            return 'LOCATION'
        elif any(org_word in base_tag.upper() for org_word in ['ORG', 'COMPANY', 'INSTITUTION']):
            return 'ORGANIZATION'
        elif any(phone_word in base_tag.upper() for phone_word in ['PHONE', 'MOBILE', 'TEL']):
            return 'PHONE'
        elif any(email_word in base_tag.upper() for email_word in ['EMAIL', 'MAIL']):
            return 'EMAIL'
        elif any(id_word in base_tag.upper() for id_word in ['ID', 'PASSPORT', 'LICENSE', 'NATIONAL']):
            return 'ID_NUMBER'
        elif any(addr_word in base_tag.upper() for addr_word in ['ADDRESS', 'STREET', 'BUILDING']):
            return 'ADDRESS'
        
        return 'OTHER'
    
    def convert_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Wojood dataset to PII format"""
        df_converted = df.copy()
        
        # Map tags to PII categories
        df_converted['pii_label'] = df_converted['tags'].apply(self.map_entity)
        
        # Keep only PII entities (remove 'O' and 'OTHER')
        pii_df = df_converted[~df_converted['pii_label'].isin(['OTHER', 'O'])].copy()
        
        return pii_df
    
    def extract_pii_entities(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract actual entity text by PII category"""
        entities_by_type = defaultdict(list)
        
        # Group by sentence and extract entity spans
        for sentence_id in df['global_sentence_id'].unique():
            sentence_df = df[df['global_sentence_id'] == sentence_id].sort_values('token_id')
            
            current_entity = None
            current_tokens = []
            current_type = None
            
            for _, row in sentence_df.iterrows():
                tag = row['tags']
                token = row['token']
                pii_type = self.map_entity(tag)
                
                if tag.startswith('B-') and pii_type != 'OTHER':
                    # Start of new entity
                    if current_entity and current_type != 'OTHER':
                        entities_by_type[current_type].append(' '.join(current_tokens))
                    current_entity = tag
                    current_tokens = [token]
                    current_type = pii_type
                elif tag.startswith('I-') and current_entity and pii_type == current_type:
                    # Continue current entity
                    current_tokens.append(token)
                else:
                    # End of entity
                    if current_entity and current_type != 'OTHER':
                        entities_by_type[current_type].append(' '.join(current_tokens))
                    current_entity = None
                    current_tokens = []
                    current_type = None
            
            # Add last entity if exists
            if current_entity and current_type != 'OTHER':
                entities_by_type[current_type].append(' '.join(current_tokens))
        
        return dict(entities_by_type)

def analyze_wojood_pii_mapping():
    """Analyze the mapping results"""
    mapper = WojoodToPIIMapper()
    
    # Load Wojood datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f'Wojood/Wojood1_1_nested/{split}.csv')
        datasets[split] = df
    
    print("WOJOOD TO PII MAPPING ANALYSIS")
    print("=" * 50)
    
    all_entities = defaultdict(list)
    
    for split_name, df in datasets.items():
        print(f"\n{split_name.upper()} Dataset Mapping:")
        print("-" * 30)
        
        # Convert dataset
        pii_df = mapper.convert_dataset(df)
        
        # Extract entities
        entities = mapper.extract_pii_entities(df)
        
        # Merge with all entities
        for pii_type, entity_list in entities.items():
            all_entities[pii_type].extend(entity_list)
        
        # Show statistics
        pii_counts = pii_df['pii_label'].value_counts()
        print(f"PII entities found: {len(pii_df)}")
        print("PII distribution:")
        for pii_type, count in pii_counts.items():
            print(f"  {pii_type}: {count}")
        
        # Show examples
        print(f"\nSample entities from {split_name}:")
        for pii_type, entity_list in entities.items():
            if entity_list:
                examples = entity_list[:3]  # Show first 3
                print(f"  {pii_type}: {examples}")
    
    print(f"\nOVERALL ENTITY STATISTICS:")
    print("-" * 30)
    for pii_type, entity_list in all_entities.items():
        unique_entities = list(set(entity_list))
        print(f"{pii_type}: {len(entity_list)} total, {len(unique_entities)} unique")
    
    return all_entities

if __name__ == "__main__":
    entities = analyze_wojood_pii_mapping()
