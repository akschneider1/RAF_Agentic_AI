
import re
import unicodedata
from typing import List, Dict, Tuple
from collections import defaultdict

class AdvancedArabicProcessor:
    """Enhanced Arabic text processing for better NER performance"""
    
    def __init__(self):
        # Extended character mappings
        self.char_mappings = {
            # Alef variations
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            # Taa variations
            'ة': 'ه', 'ت': 'ت',
            # Yaa variations
            'ي': 'ى', 'ئ': 'ى', 'ؤ': 'و',
            # Haa variations
            'ه': 'ه', 'ح': 'ح',
        }
        
        # Arabic diacritics (comprehensive)
        self.diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]')
        
        # Arabic punctuation normalization
        self.punctuation_map = {
            '؟': '?',
            '،': ',',
            '؛': ';',
            '٪': '%',
            '٫': ',',
            '٬': ',',
        }
        
        # Named entity indicators in Arabic
        self.person_indicators = [
            'الأستاذ', 'الدكتور', 'المهندس', 'الشيخ', 'السيد', 'السيدة',
            'أستاذ', 'دكتور', 'مهندس', 'شيخ', 'سيد', 'سيدة'
        ]
        
        self.location_indicators = [
            'مدينة', 'محافظة', 'منطقة', 'حي', 'شارع', 'طريق', 'جادة',
            'المملكة', 'دولة', 'إمارة', 'ولاية'
        ]
        
        self.org_indicators = [
            'شركة', 'مؤسسة', 'منظمة', 'جامعة', 'معهد', 'مستشفى',
            'وزارة', 'هيئة', 'مجلس', 'لجنة'
        ]
    
    def normalize_text(self, text: str) -> str:
        """Advanced Arabic text normalization"""
        if not text:
            return text
        
        # Remove diacritics
        text = self.diacritics.sub('', text)
        
        # Normalize characters
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)
        
        # Normalize punctuation
        for old_punct, new_punct in self.punctuation_map.items():
            text = text.replace(old_punct, new_punct)
        
        # Normalize whitespace and repeated characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Limit repeated chars to 2
        
        # Handle mixed scripts (Arabic + Latin)
        text = self._normalize_mixed_script(text)
        
        return text.strip()
    
    def _normalize_mixed_script(self, text: str) -> str:
        """Handle Arabic-English mixed text"""
        # Add spaces around script transitions
        text = re.sub(r'([\u0600-\u06FF])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])([\u0600-\u06FF])', r'\1 \2', text)
        return text
    
    def extract_contextual_features(self, tokens: List[str], index: int) -> Dict[str, bool]:
        """Extract contextual features for better NER"""
        features = {}
        token = tokens[index].lower()
        
        # Position features
        features['is_first'] = index == 0
        features['is_last'] = index == len(tokens) - 1
        
        # Morphological features
        features['has_arabic'] = bool(re.search(r'[\u0600-\u06FF]', token))
        features['has_digits'] = bool(re.search(r'\d', token))
        features['has_punctuation'] = bool(re.search(r'[^\w\s]', token))
        features['is_capitalized'] = token[0].isupper() if token else False
        
        # Context features
        if index > 0:
            prev_token = tokens[index - 1].lower()
            features['prev_is_person_indicator'] = prev_token in self.person_indicators
            features['prev_is_location_indicator'] = prev_token in self.location_indicators
            features['prev_is_org_indicator'] = prev_token in self.org_indicators
        
        if index < len(tokens) - 1:
            next_token = tokens[index + 1].lower()
            features['next_is_person_indicator'] = next_token in self.person_indicators
            features['next_is_location_indicator'] = next_token in self.location_indicators
            features['next_is_org_indicator'] = next_token in self.org_indicators
        
        # Pattern features
        features['looks_like_phone'] = bool(re.match(r'^[\+\d\s\-\(\)]+$', token))
        features['looks_like_email'] = '@' in token
        features['looks_like_id'] = bool(re.match(r'^\d{8,15}$', token))
        
        return features
    
    def segment_arabic_text(self, text: str) -> List[str]:
        """Improved Arabic text segmentation"""
        # Handle Arabic-specific tokenization
        text = re.sub(r'([\.!\?؟])', r' \1 ', text)
        text = re.sub(r'([،؛])', r' \1 ', text)
        text = re.sub(r'(\d+)', r' \1 ', text)
        
        # Split and clean tokens
        tokens = text.split()
        cleaned_tokens = []
        
        for token in tokens:
            # Remove empty tokens
            if not token.strip():
                continue
            
            # Handle attached prepositions/conjunctions
            if len(token) > 2 and token.startswith(('و', 'ف', 'ب', 'ل', 'ك')):
                prefix = token[0]
                rest = token[1:]
                if len(rest) > 1:  # Only split if remaining part is meaningful
                    cleaned_tokens.extend([prefix, rest])
                else:
                    cleaned_tokens.append(token)
            else:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
