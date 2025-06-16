
#!/usr/bin/env python3
"""
Advanced data augmentation with context-aware PII generation and hard negative mining
"""

import pandas as pd
import random
from typing import List, Dict, Any
import re
from arabic_processor import ArabicProcessor
import numpy as np

class AdvancedPIIAugmentation:
    """Advanced augmentation techniques for PII detection"""
    
    def __init__(self):
        self.arabic_processor = ArabicProcessor()
        
        # Contextual templates for better PII detection
        self.contextual_templates = {
            'formal_documents': [
                "المواطن {PERSON} حامل الهوية رقم {ID_NUMBER} المولود في {LOCATION}",
                "يتشرف {ORGANIZATION} بدعوة السيد {PERSON} للحضور",
                "العنوان: {ADDRESS} هاتف: {PHONE} البريد: {EMAIL}",
            ],
            'informal_communication': [
                "أخي {PERSON} اتصل بي على {PHONE}",
                "راسلني على {EMAIL} وإلا كلمني {PHONE}",
                "أنا من {LOCATION} وأعمل في {ORGANIZATION}",
            ],
            'business_context': [
                "شركة {ORGANIZATION} تقع في {ADDRESS}",
                "للتواصل مع {PERSON} مدير {ORGANIZATION} على {PHONE}",
                "رقم الترخيص التجاري {ID_NUMBER} للشركة في {LOCATION}",
            ],
            'mixed_language': [
                "My name is {PERSON} and I live in {LOCATION}",
                "Contact {PERSON} at {EMAIL} or call {PHONE}",
                "Visit our office at {ADDRESS} في {LOCATION}",
            ]
        }
        
        # Hard negative examples (challenging cases)
        self.hard_negatives = {
            'false_positives': [
                "رقم الصفحة 123456789",  # Not an ID
                "العام 2023 في الشهر 05",  # Not a phone
                "البريد السعودي صندوق 123",  # Not email
            ],
            'partial_pii': [
                "أحمد... (اسم محذوف)",  # Incomplete name
                "05xxxxxxxx",  # Masked phone
                "user@domain.",  # Incomplete email
            ],
            'boundary_cases': [
                "د. أحمد محمد الطبيب",  # Title + name
                "0501234567 ext 123",  # Phone with extension
                "ahmed.mohammed@company.com.sa",  # Complex email
            ]
        }
    
    def generate_contextual_examples(self, num_examples: int = 5000) -> List[Dict[str, Any]]:
        """Generate contextually rich PII examples"""
        examples = []
        
        for i in range(num_examples):
            # Choose context type
            context_type = random.choice(list(self.contextual_templates.keys()))
            template = random.choice(self.contextual_templates[context_type])
            
            # Generate realistic PII values
            pii_values = self._generate_realistic_pii_values(context_type)
            
            # Fill template
            text = template
            detected_pii = []
            
            for pii_type, value in pii_values.items():
                placeholder = f"{{{pii_type}}}"
                if placeholder in text:
                    start_pos = text.find(placeholder)
                    text = text.replace(placeholder, value, 1)
                    
                    detected_pii.append({
                        'text': value,
                        'type': pii_type,
                        'start': start_pos,
                        'end': start_pos + len(value),
                        'context': context_type
                    })
            
            examples.append({
                'sentence_id': f"contextual_{i}",
                'text': text,
                'pii_entities': detected_pii,
                'context_type': context_type,
                'source': 'contextual_augmentation'
            })
        
        return examples
    
    def generate_hard_negatives(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate hard negative examples to improve precision"""
        examples = []
        
        for i in range(num_examples):
            category = random.choice(list(self.hard_negatives.keys()))
            base_text = random.choice(self.hard_negatives[category])
            
            # Add more context to make it realistic
            contexts = [
                f"في الوثيقة المرفقة: {base_text}",
                f"كما هو موضح في {base_text}",
                f"يرجى ملاحظة أن {base_text}",
            ]
            
            full_text = random.choice(contexts)
            
            examples.append({
                'sentence_id': f"hard_negative_{i}",
                'text': full_text,
                'pii_entities': [],  # No PII entities
                'context_type': f"hard_negative_{category}",
                'source': 'hard_negative_augmentation'
            })
        
        return examples
    
    def generate_adversarial_examples(self, base_examples: List[Dict], num_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate adversarial examples by modifying existing ones"""
        adversarial_examples = []
        
        for example in base_examples[:500]:  # Take subset for adversarial generation
            for variant in range(num_variants):
                modified_text = example['text']
                modified_pii = example['pii_entities'].copy()
                
                # Apply different adversarial transformations
                transformation = random.choice([
                    'add_noise',
                    'partial_occlusion',
                    'format_variation',
                    'context_shift'
                ])
                
                if transformation == 'add_noise':
                    # Add Arabic diacritics randomly
                    modified_text = self._add_arabic_diacritics(modified_text)
                
                elif transformation == 'partial_occlusion':
                    # Partially hide PII with asterisks
                    modified_text, modified_pii = self._partial_occlusion(modified_text, modified_pii)
                
                elif transformation == 'format_variation':
                    # Change PII formatting
                    modified_text, modified_pii = self._format_variation(modified_text, modified_pii)
                
                elif transformation == 'context_shift':
                    # Add confusing context
                    modified_text = self._add_confusing_context(modified_text)
                
                adversarial_examples.append({
                    'sentence_id': f"adversarial_{example['sentence_id']}_{variant}",
                    'text': modified_text,
                    'pii_entities': modified_pii,
                    'context_type': f"adversarial_{transformation}",
                    'source': 'adversarial_augmentation'
                })
        
        return adversarial_examples
    
    def _generate_realistic_pii_values(self, context_type: str) -> Dict[str, str]:
        """Generate realistic PII values based on context"""
        pii_values = {}
        
        # Arabic names based on context
        if context_type == 'formal_documents':
            names = ['عبدالرحمن بن سالم الغامدي', 'فاطمة أحمد المالكي', 'محمد عبدالله القحطاني']
        else:
            names = ['أحمد', 'فاطمة', 'محمد', 'عائشة', 'علي']
        
        pii_values['PERSON'] = random.choice(names)
        
        # Context-appropriate locations
        if context_type == 'business_context':
            locations = ['الرياض، حي العليا', 'جدة، شارع التحلية', 'الدمام، الكورنيش']
        else:
            locations = ['الرياض', 'جدة', 'مكة', 'المدينة', 'الدمام']
        
        pii_values['LOCATION'] = random.choice(locations)
        
        # Generate other PII types
        pii_values['PHONE'] = f"05{random.randint(10000000, 99999999)}"
        pii_values['EMAIL'] = f"{random.choice(['ahmed', 'fatima', 'mohammed'])}@{random.choice(['gmail.com', 'hotmail.com', 'company.com.sa'])}"
        pii_values['ID_NUMBER'] = f"{random.randint(1000000000, 2999999999)}"
        pii_values['ORGANIZATION'] = random.choice(['شركة المملكة', 'مؤسسة النور', 'البنك الأهلي'])
        pii_values['ADDRESS'] = f"شارع {random.choice(['الملك فهد', 'العليا', 'التحلية'])} رقم {random.randint(1, 999)}"
        
        return pii_values
    
    def _add_arabic_diacritics(self, text: str) -> str:
        """Add Arabic diacritics to make recognition harder"""
        diacritics = ['َ', 'ِ', 'ُ', 'ّ', 'ْ']
        words = text.split()
        
        for i, word in enumerate(words):
            if random.random() < 0.3 and any('\u0600' <= c <= '\u06FF' for c in word):
                # Add diacritics to Arabic words
                modified_word = ""
                for char in word:
                    modified_word += char
                    if '\u0600' <= char <= '\u06FF' and random.random() < 0.2:
                        modified_word += random.choice(diacritics)
                words[i] = modified_word
        
        return ' '.join(words)
    
    def _partial_occlusion(self, text: str, pii_entities: List[Dict]) -> tuple:
        """Partially hide PII entities"""
        modified_text = text
        modified_pii = []
        
        # Sort by position in reverse to maintain indices
        sorted_entities = sorted(pii_entities, key=lambda x: x['start'], reverse=True)
        
        for entity in sorted_entities:
            if random.random() < 0.5:  # 50% chance to occlude
                entity_text = entity['text']
                if len(entity_text) > 3:
                    # Replace middle characters with asterisks
                    occluded = entity_text[:2] + '*' * (len(entity_text) - 4) + entity_text[-2:]
                    modified_text = modified_text[:entity['start']] + occluded + modified_text[entity['end']:]
                    
                    # Update entity
                    entity['text'] = occluded
                    modified_pii.append(entity)
                else:
                    modified_pii.append(entity)
            else:
                modified_pii.append(entity)
        
        return modified_text, modified_pii
    
    def create_enhanced_augmented_dataset(self) -> pd.DataFrame:
        """Create comprehensive augmented dataset"""
        print("CREATING ENHANCED AUGMENTED DATASET")
        print("=" * 50)
        
        all_examples = []
        
        # 1. Contextual examples
        print("Generating contextual examples...")
        contextual_examples = self.generate_contextual_examples(5000)
        all_examples.extend(contextual_examples)
        
        # 2. Hard negatives
        print("Generating hard negative examples...")
        hard_negative_examples = self.generate_hard_negatives(1000)
        all_examples.extend(hard_negative_examples)
        
        # 3. Adversarial examples
        print("Generating adversarial examples...")
        adversarial_examples = self.generate_adversarial_examples(contextual_examples, 2)
        all_examples.extend(adversarial_examples)
        
        # Convert to token-level format
        print("Converting to token-level format...")
        token_records = self._convert_to_token_format(all_examples)
        
        # Combine with original Wojood data
        print("Loading original Wojood data...")
        wojood_df = pd.read_csv('Wojood/Wojood1_1_nested/train.csv')
        if 'sentence_id' not in wojood_df.columns:
            wojood_df['sentence_id'] = wojood_df['global_sentence_id']
        
        # Combine datasets
        enhanced_df = pd.concat([wojood_df, token_records], ignore_index=True)
        enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Enhanced dataset created:")
        print(f"  Original Wojood: {len(wojood_df)} tokens")
        print(f"  Augmented data: {len(token_records)} tokens")
        print(f"  Total: {len(enhanced_df)} tokens")
        
        # Save enhanced dataset
        enhanced_df.to_csv('train_enhanced_augmented.csv', index=False)
        
        return enhanced_df

def main():
    """Create enhanced augmented dataset"""
    augmenter = AdvancedPIIAugmentation()
    enhanced_df = augmenter.create_enhanced_augmented_dataset()
    
    print("\n✅ Enhanced augmented dataset created!")
    print("File saved: train_enhanced_augmented.csv")

if __name__ == "__main__":
    main()
