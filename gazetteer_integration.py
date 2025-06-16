
#!/usr/bin/env python3
"""
Gazetteer Integration Module
Integrates Jordan-specific gazetteers with the existing PII detection system
"""

import json
import csv
import os
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from rules import PIIDetector, PIIMatch
import re
from collections import defaultdict

@dataclass
class GazetteerMatch:
    """Match found using gazetteer lookup"""
    text: str
    pii_type: str
    start_pos: int
    end_pos: int
    confidence: float
    gazetteer_source: str
    subcategory: str

class GazetteerEnhancedDetector:
    """Enhanced PII detector that uses gazetteers alongside rule-based detection"""
    
    def __init__(self, gazetteer_dir: str = "jordan_gazetteers"):
        self.base_detector = PIIDetector()
        self.gazetteers = {}
        self.gazetteer_patterns = {}
        self.load_gazetteers(gazetteer_dir)
        self._compile_gazetteer_patterns()
    
    def load_gazetteers(self, gazetteer_dir: str):
        """Load all gazetteer files"""
        if not os.path.exists(gazetteer_dir):
            print(f"Warning: Gazetteer directory {gazetteer_dir} not found")
            return
        
        print(f"Loading gazetteers from {gazetteer_dir}/")
        
        gazetteer_files = [
            'jordan_person.json',
            'jordan_location.json', 
            'jordan_organization.json',
            'jordan_phone.json'
        ]
        
        for filename in gazetteer_files:
            filepath = os.path.join(gazetteer_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    category = filename.replace('jordan_', '').replace('.json', '').upper()
                    self.gazetteers[category] = data
                    
                    print(f"  ✅ Loaded {len(data)} entries for {category}")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
            else:
                print(f"  ⚠️  File not found: {filename}")
    
    def _compile_gazetteer_patterns(self):
        """Compile gazetteer entries into efficient search patterns"""
        print("Compiling gazetteer patterns...")
        
        for category, entries in self.gazetteers.items():
            # Group by confidence level for priority matching
            high_conf_entries = []
            medium_conf_entries = []
            low_conf_entries = []
            
            for entry in entries:
                confidence = entry.get('confidence', 0.5)
                text = entry['text'].strip()
                
                if len(text) < 2:  # Skip very short entries
                    continue
                
                if confidence >= 0.8:
                    high_conf_entries.append(entry)
                elif confidence >= 0.6:
                    medium_conf_entries.append(entry)
                else:
                    low_conf_entries.append(entry)
            
            # Create regex patterns for each confidence level
            self.gazetteer_patterns[category] = {
                'high': self._create_pattern_group(high_conf_entries),
                'medium': self._create_pattern_group(medium_conf_entries),
                'low': self._create_pattern_group(low_conf_entries)
            }
            
            total_patterns = len(high_conf_entries) + len(medium_conf_entries) + len(low_conf_entries)
            print(f"  {category}: {total_patterns} patterns compiled")
    
    def _create_pattern_group(self, entries: List[Dict]) -> Dict:
        """Create regex pattern group from entries"""
        if not entries:
            return {'pattern': None, 'entries': []}
        
        # Sort by length (longest first) to prioritize longer matches
        entries = sorted(entries, key=lambda x: len(x['text']), reverse=True)
        
        # Escape special regex characters and create pattern
        escaped_texts = []
        for entry in entries:
            text = re.escape(entry['text'])
            escaped_texts.append(text)
        
        # Create pattern with word boundaries
        pattern_str = r'\b(?:' + '|'.join(escaped_texts) + r')\b'
        
        try:
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            return {'pattern': compiled_pattern, 'entries': entries}
        except re.error as e:
            print(f"Warning: Regex compilation error: {e}")
            return {'pattern': None, 'entries': entries}
    
    def find_gazetteer_matches(self, text: str) -> List[GazetteerMatch]:
        """Find PII matches using gazetteers"""
        matches = []
        
        for category, pattern_groups in self.gazetteer_patterns.items():
            # Check each confidence level (high to low)
            for conf_level, pattern_group in pattern_groups.items():
                if pattern_group['pattern'] is None:
                    continue
                
                # Find all matches
                for match in pattern_group['pattern'].finditer(text):
                    matched_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Find the specific entry that matched
                    matched_entry = None
                    for entry in pattern_group['entries']:
                        if entry['text'].lower() == matched_text.lower():
                            matched_entry = entry
                            break
                    
                    if matched_entry:
                        # Adjust confidence based on context
                        base_confidence = matched_entry.get('confidence', 0.5)
                        context_confidence = self._analyze_context_confidence(text, start_pos, end_pos, category)
                        final_confidence = min(0.95, base_confidence * context_confidence)
                        
                        matches.append(GazetteerMatch(
                            text=matched_text,
                            pii_type=category,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=final_confidence,
                            gazetteer_source=matched_entry.get('source', 'unknown'),
                            subcategory=matched_entry.get('subcategory', 'general')
                        ))
        
        # Remove overlapping matches (keep highest confidence)
        return self._resolve_overlaps(matches)
    
    def _analyze_context_confidence(self, text: str, start_pos: int, end_pos: int, category: str) -> float:
        """Analyze surrounding context to adjust confidence"""
        # Extract context around the match
        context_size = 50
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        confidence_modifier = 1.0
        
        # Context indicators for different PII types
        if category == 'PERSON':
            positive_indicators = ['اسم', 'السيد', 'الأستاذ', 'الدكتور', 'المهندس', 'يدعى', 'المدعو']
            negative_indicators = ['شركة', 'مؤسسة', 'جامعة', 'وزارة', 'مدينة', 'محافظة']
            
        elif category == 'LOCATION':
            positive_indicators = ['في', 'إلى', 'من', 'مدينة', 'محافظة', 'حي', 'شارع', 'طريق']
            negative_indicators = ['اسم', 'يدعى', 'شخص']
            
        elif category == 'ORGANIZATION':
            positive_indicators = ['شركة', 'مؤسسة', 'جامعة', 'وزارة', 'هيئة', 'مجلس', 'بنك']
            negative_indicators = ['اسم', 'شخص', 'يدعى']
            
        else:
            return confidence_modifier
        
        # Check for positive indicators
        for indicator in positive_indicators:
            if indicator in context:
                confidence_modifier *= 1.2
                break
        
        # Check for negative indicators
        for indicator in negative_indicators:
            if indicator in context:
                confidence_modifier *= 0.7
                break
        
        return min(1.0, confidence_modifier)
    
    def _resolve_overlaps(self, matches: List[GazetteerMatch]) -> List[GazetteerMatch]:
        """Resolve overlapping matches by keeping highest confidence"""
        if not matches:
            return matches
        
        # Sort by position
        matches = sorted(matches, key=lambda x: (x.start_pos, x.end_pos))
        
        resolved = []
        
        for match in matches:
            # Check if this match overlaps with any already resolved match
            overlaps = False
            
            for existing in resolved:
                if (match.start_pos < existing.end_pos and 
                    match.end_pos > existing.start_pos):
                    # Overlap detected - keep the higher confidence match
                    if match.confidence > existing.confidence:
                        resolved.remove(existing)
                        resolved.append(match)
                    overlaps = True
                    break
            
            if not overlaps:
                resolved.append(match)
        
        return resolved
    
    def detect_enhanced_pii(self, text: str, min_confidence: float = 0.7) -> Tuple[List[PIIMatch], List[GazetteerMatch]]:
        """Detect PII using both rule-based and gazetteer methods"""
        # Get rule-based matches
        rule_matches = self.base_detector.detect_all_pii(text, min_confidence)
        
        # Get gazetteer matches
        gazetteer_matches = self.find_gazetteer_matches(text)
        
        # Filter gazetteer matches by confidence
        filtered_gazetteer = [m for m in gazetteer_matches if m.confidence >= min_confidence]
        
        return rule_matches, filtered_gazetteer
    
    def combine_all_matches(self, text: str, min_confidence: float = 0.7) -> List[Dict]:
        """Combine rule-based and gazetteer matches into unified format"""
        rule_matches, gazetteer_matches = self.detect_enhanced_pii(text, min_confidence)
        
        combined_matches = []
        
        # Add rule-based matches
        for match in rule_matches:
            combined_matches.append({
                'text': match.text,
                'type': match.pii_type,
                'start': match.start_pos,
                'end': match.end_pos,
                'confidence': match.confidence,
                'source': 'rule_based',
                'pattern': match.pattern_name
            })
        
        # Add gazetteer matches
        for match in gazetteer_matches:
            combined_matches.append({
                'text': match.text,
                'type': match.pii_type,
                'start': match.start_pos,
                'end': match.end_pos,
                'confidence': match.confidence,
                'source': 'gazetteer',
                'gazetteer_source': match.gazetteer_source,
                'subcategory': match.subcategory
            })
        
        # Remove overlaps between rule-based and gazetteer matches
        return self._resolve_combined_overlaps(combined_matches)
    
    def _resolve_combined_overlaps(self, matches: List[Dict]) -> List[Dict]:
        """Resolve overlaps between different detection methods"""
        if not matches:
            return matches
        
        # Sort by position
        matches = sorted(matches, key=lambda x: (x['start'], x['end']))
        
        resolved = []
        
        for match in matches:
            overlaps = False
            
            for i, existing in enumerate(resolved):
                if (match['start'] < existing['end'] and 
                    match['end'] > existing['start']):
                    # Overlap detected
                    overlaps = True
                    
                    # Prefer gazetteer matches for names and locations if high confidence
                    if (match['source'] == 'gazetteer' and 
                        match['type'] in ['PERSON', 'LOCATION'] and 
                        match['confidence'] > 0.8):
                        resolved[i] = match
                    # Otherwise prefer higher confidence
                    elif match['confidence'] > existing['confidence']:
                        resolved[i] = match
                    
                    break
            
            if not overlaps:
                resolved.append(match)
        
        return resolved

def test_gazetteer_integration():
    """Test the gazetteer integration"""
    print("🧪 TESTING GAZETTEER INTEGRATION")
    print("=" * 50)
    
    # Create detector
    detector = GazetteerEnhancedDetector()
    
    # Test cases
    test_texts = [
        "السيد أحمد العبدالله يعمل في الجامعة الأردنية في عمان",
        "اتصل بالدكتور محمد الزعبي على رقم 077123456",
        "تقع شركة البنك الأهلي الأردني في الشميساني",
        "سافر خالد المجالي إلى محافظة إربد الأسبوع الماضي",
        "للاستفسار اتصل على +962 77 123 4567"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 40)
        
        # Get all matches
        matches = detector.combine_all_matches(text, min_confidence=0.6)
        
        print(f"Found {len(matches)} PII entities:")
        for match in matches:
            print(f"  • {match['type']}: '{match['text']}' "
                  f"(conf: {match['confidence']:.2f}, source: {match['source']})")

if __name__ == "__main__":
    test_gazetteer_integration()
