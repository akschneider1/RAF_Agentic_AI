
#!/usr/bin/env python3
"""
Jordan Gazetteer Scraper
Creates comprehensive gazetteers from publicly available Jordanian datasets
to enhance PII detection accuracy for Jordanian entities
"""

import requests
import json
import csv
import re
import time
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from bs4 import BeautifulSoup
import os

@dataclass
class GazetteerEntry:
    """Single gazetteer entry"""
    text: str
    category: str
    subcategory: str
    source: str
    confidence: float = 1.0
    metadata: Optional[Dict] = None

class JordanGazetteerScraper:
    """Scrapes and creates Jordan-specific gazetteers for PII detection"""
    
    def __init__(self):
        self.gazetteers = {
            'PERSON': defaultdict(list),
            'LOCATION': defaultdict(list),
            'ORGANIZATION': defaultdict(list),
            'PHONE': defaultdict(list),
            'ID_NUMBER': defaultdict(list)
        }
        
        # Jordan-specific patterns and sources
        self.jordan_sources = {
            'government': [
                'https://www.jordan.gov.jo',
                'https://dos.gov.jo',  # Department of Statistics
                'https://www.cbj.gov.jo'  # Central Bank of Jordan
            ],
            'universities': [
                'University of Jordan', 'Jordan University of Science and Technology',
                'Yarmouk University', 'Mu\'tah University', 'Al-Balqa Applied University',
                'Hashemite University', 'Al al-Bayt University', 'Philadelphia University',
                'Applied Science Private University', 'Princess Sumaya University'
            ],
            'cities_governorates': [
                'عمان', 'إربد', 'الزرقاء', 'المفرق', 'جرش', 'عجلون', 'البلقاء', 'مادبا',
                'الكرك', 'الطفيلة', 'معان', 'العقبة'
            ]
        }
        
        # Common Jordanian name patterns
        self.jordan_name_patterns = {
            'male_first': [
                'محمد', 'أحمد', 'خالد', 'عبدالله', 'عمر', 'يوسف', 'عبدالرحمن', 'حسام',
                'طارق', 'سامر', 'وليد', 'نادر', 'باسم', 'مازن', 'فادي', 'رامي',
                'عماد', 'إياد', 'مؤيد', 'معاذ', 'زياد', 'جهاد', 'نبيل', 'وسام'
            ],
            'female_first': [
                'فاطمة', 'عائشة', 'خديجة', 'زينب', 'مريم', 'سارة', 'نور', 'رنا',
                'هند', 'ليلى', 'أسماء', 'آمنة', 'سعاد', 'منى', 'رنيم', 'دينا',
                'لينا', 'رغد', 'شذى', 'ندى', 'ريم', 'هبة', 'نايا', 'سلمى'
            ],
            'family_names': [
                'العبدالله', 'المحمد', 'الأحمد', 'الخطيب', 'النجار', 'الزعبي', 'العجارمة',
                'البطاينة', 'الخوالدة', 'الطوالبة', 'القضاة', 'الشوابكة', 'المجالي',
                'الفايز', 'الزيود', 'الحباشنة', 'الحوراني', 'الكايد', 'العموش',
                'الصمادي', 'الربابعة', 'العنانزة', 'الصرايرة', 'الحجازين'
            ]
        }
        
        # Jordan phone number patterns
        self.jordan_phone_patterns = [
            r'\+962\s*7[789]\s*\d{7}',  # Mobile
            r'07[789]\d{7}',            # Local mobile
            r'\+962\s*[2-6]\s*\d{7}',   # Landline
            r'0[2-6]\d{7}'              # Local landline
        ]

    def scrape_jordan_locations(self) -> List[GazetteerEntry]:
        """Scrape Jordan locations from various sources"""
        locations = []
        
        # Governorates and major cities
        governorates = [
            ('عمان', 'محافظة عمان'), ('إربد', 'محافظة إربد'), ('الزرقاء', 'محافظة الزرقاء'),
            ('المفرق', 'محافظة المفرق'), ('جرش', 'محافظة جرش'), ('عجلون', 'محافظة عجلون'),
            ('البلقاء', 'محافظة البلقاء'), ('مادبا', 'محافظة مادبا'), ('الكرك', 'محافظة الكرك'),
            ('الطفيلة', 'محافظة الطفيلة'), ('معان', 'محافظة معان'), ('العقبة', 'محافظة العقبة')
        ]
        
        for city, governorate in governorates:
            locations.extend([
                GazetteerEntry(city, 'LOCATION', 'city', 'jordan_official', 1.0),
                GazetteerEntry(governorate, 'LOCATION', 'governorate', 'jordan_official', 1.0)
            ])
        
        # Major districts in Amman
        amman_districts = [
            'الدوار الثالث', 'الدوار الرابع', 'الدوار الخامس', 'الدوار السادس',
            'جبل عمان', 'جبل اللويبدة', 'جبل الحسين', 'جبل النزهة', 'جبل التاج',
            'الصويفية', 'الشميساني', 'عبدون', 'الرابية', 'خلدا', 'مرج الحمام',
            'طبربور', 'النصر', 'الوحدات', 'بسمان', 'ماركا'
        ]
        
        for district in amman_districts:
            locations.append(
                GazetteerEntry(district, 'LOCATION', 'district', 'jordan_amman', 0.9)
            )
        
        # Palestinian refugee camps in Jordan
        refugee_camps = [
            'مخيم الوحدات', 'مخيم البقعة', 'مخيم الحسين', 'مخيم جرش',
            'مخيم الطالبية', 'مخيم مادبا', 'مخيم الزرقاء', 'مخيم إربد',
            'مخيم عين الباشا', 'مخيم الأزرق'
        ]
        
        for camp in refugee_camps:
            locations.append(
                GazetteerEntry(camp, 'LOCATION', 'refugee_camp', 'jordan_unrwa', 0.8)
            )
        
        return locations

    def generate_jordan_names(self) -> List[GazetteerEntry]:
        """Generate comprehensive Jordan name combinations"""
        names = []
        
        # Generate full names (first + family)
        for first_name in self.jordan_name_patterns['male_first'][:20]:  # Top 20
            for family_name in self.jordan_name_patterns['family_names'][:15]:  # Top 15
                full_name = f"{first_name} {family_name}"
                names.append(
                    GazetteerEntry(full_name, 'PERSON', 'male_full_name', 'jordan_generated', 0.8)
                )
        
        for first_name in self.jordan_name_patterns['female_first'][:20]:
            for family_name in self.jordan_name_patterns['family_names'][:15]:
                full_name = f"{first_name} {family_name}"
                names.append(
                    GazetteerEntry(full_name, 'PERSON', 'female_full_name', 'jordan_generated', 0.8)
                )
        
        # Add standalone first names
        for first_name in self.jordan_name_patterns['male_first']:
            names.append(
                GazetteerEntry(first_name, 'PERSON', 'male_first_name', 'jordan_common', 0.7)
            )
        
        for first_name in self.jordan_name_patterns['female_first']:
            names.append(
                GazetteerEntry(first_name, 'PERSON', 'female_first_name', 'jordan_common', 0.7)
            )
        
        # Add family names
        for family_name in self.jordan_name_patterns['family_names']:
            names.append(
                GazetteerEntry(family_name, 'PERSON', 'family_name', 'jordan_tribes', 0.6)
            )
        
        return names

    def scrape_jordan_organizations(self) -> List[GazetteerEntry]:
        """Create Jordan organization gazetteer"""
        organizations = []
        
        # Government ministries
        ministries = [
            'وزارة الداخلية', 'وزارة الخارجية', 'وزارة المالية', 'وزارة التربية والتعليم',
            'وزارة الصحة', 'وزارة العمل', 'وزارة الزراعة', 'وزارة الطاقة',
            'وزارة النقل', 'وزارة السياحة', 'وزارة الثقافة', 'وزارة الشباب',
            'وزارة التعليم العالي', 'وزارة البيئة', 'وزارة التنمية الاجتماعية'
        ]
        
        for ministry in ministries:
            organizations.append(
                GazetteerEntry(ministry, 'ORGANIZATION', 'government_ministry', 'jordan_official', 1.0)
            )
        
        # Universities
        universities = [
            'الجامعة الأردنية', 'جامعة اليرموك', 'جامعة العلوم والتكنولوجيا',
            'جامعة مؤتة', 'جامعة البلقاء التطبيقية', 'الجامعة الهاشمية',
            'جامعة آل البيت', 'جامعة فيلادلفيا', 'جامعة العلوم التطبيقية',
            'جامعة الأميرة سمية للتكنولوجيا', 'الجامعة الألمانية الأردنية'
        ]
        
        for university in universities:
            organizations.append(
                GazetteerEntry(university, 'ORGANIZATION', 'university', 'jordan_education', 0.9)
            )
        
        # Major companies and banks
        companies = [
            'البنك الأهلي الأردني', 'بنك الإسكان', 'البنك العربي', 'بنك القاهرة عمان',
            'البنك الإسلامي الأردني', 'مجموعة زين', 'أورانج الأردن', 'أمنية',
            'الخطوط الجوية الأردنية', 'مصفاة البترول الأردنية', 'شركة الكهرباء الوطنية',
            'مجموعة نور الدين', 'شركة الفوسفات الأردنية', 'شركة البوتاس العربية'
        ]
        
        for company in companies:
            organizations.append(
                GazetteerEntry(company, 'ORGANIZATION', 'company', 'jordan_business', 0.8)
            )
        
        return organizations

    def generate_jordan_phones(self) -> List[GazetteerEntry]:
        """Generate Jordan phone number patterns"""
        phones = []
        
        # Mobile number prefixes in Jordan
        mobile_prefixes = ['077', '078', '079']
        
        # Generate sample phone numbers for pattern recognition
        for prefix in mobile_prefixes:
            for i in range(0, 10):  # Generate 10 examples per prefix
                sample_number = f"{prefix}{str(i).zfill(7)}"
                phones.append(
                    GazetteerEntry(sample_number, 'PHONE', 'mobile', 'jordan_telecom', 0.7)
                )
                
                # Also add with country code
                international = f"+962 {sample_number[1:]}"
                phones.append(
                    GazetteerEntry(international, 'PHONE', 'mobile_international', 'jordan_telecom', 0.7)
                )
        
        # Landline patterns (Amman area code 06)
        amman_samples = ['065551234', '065552345', '065553456']
        for number in amman_samples:
            phones.extend([
                GazetteerEntry(number, 'PHONE', 'landline_amman', 'jordan_telecom', 0.6),
                GazetteerEntry(f"+962 {number[1:]}", 'PHONE', 'landline_international', 'jordan_telecom', 0.6)
            ])
        
        return phones

    def scrape_web_sources(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Safely scrape web content with retries"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    return response.text
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Error scraping {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None

    def create_all_gazetteers(self) -> Dict[str, List[GazetteerEntry]]:
        """Create all Jordan-specific gazetteers"""
        print("🇯🇴 CREATING JORDAN GAZETTEERS")
        print("=" * 50)
        
        all_gazetteers = {}
        
        # Generate locations
        print("📍 Creating location gazetteer...")
        locations = self.scrape_jordan_locations()
        all_gazetteers['LOCATION'] = locations
        print(f"   ✅ Created {len(locations)} location entries")
        
        # Generate names
        print("👤 Creating person name gazetteer...")
        names = self.generate_jordan_names()
        all_gazetteers['PERSON'] = names
        print(f"   ✅ Created {len(names)} person name entries")
        
        # Generate organizations
        print("🏢 Creating organization gazetteer...")
        organizations = self.scrape_jordan_organizations()
        all_gazetteers['ORGANIZATION'] = organizations
        print(f"   ✅ Created {len(organizations)} organization entries")
        
        # Generate phone patterns
        print("📞 Creating phone number gazetteer...")
        phones = self.generate_jordan_phones()
        all_gazetteers['PHONE'] = phones
        print(f"   ✅ Created {len(phones)} phone number entries")
        
        return all_gazetteers

    def save_gazetteers(self, gazetteers: Dict[str, List[GazetteerEntry]], output_dir: str = "jordan_gazetteers"):
        """Save gazetteers to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving gazetteers to {output_dir}/")
        
        for category, entries in gazetteers.items():
            # Save as CSV
            csv_file = os.path.join(output_dir, f"jordan_{category.lower()}.csv")
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'category', 'subcategory', 'source', 'confidence', 'metadata'])
                
                for entry in entries:
                    writer.writerow([
                        entry.text, entry.category, entry.subcategory, 
                        entry.source, entry.confidence, 
                        json.dumps(entry.metadata) if entry.metadata else ''
                    ])
            
            # Save as JSON for easy loading
            json_file = os.path.join(output_dir, f"jordan_{category.lower()}.json")
            
            json_data = []
            for entry in entries:
                json_data.append({
                    'text': entry.text,
                    'category': entry.category,
                    'subcategory': entry.subcategory,
                    'source': entry.source,
                    'confidence': entry.confidence,
                    'metadata': entry.metadata
                })
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"   📁 {category}: {len(entries)} entries → {csv_file}, {json_file}")

    def create_training_augmentation_data(self, gazetteers: Dict[str, List[GazetteerEntry]]) -> List[Dict]:
        """Create training data using gazetteers"""
        training_sentences = []
        
        # Template patterns for creating training sentences
        templates = {
            'PERSON': [
                "تم تعيين {person} في منصب جديد",
                "التقى الوزير مع {person} في العاصمة",
                "أعلن {person} عن بدء المشروع الجديد",
                "شارك {person} في المؤتمر الدولي",
                "فاز {person} بجائزة التميز"
            ],
            'LOCATION': [
                "انعقد المؤتمر في {location}",
                "سافر الوفد إلى {location}",
                "تقع الشركة في {location}",
                "افتتح المركز الجديد في {location}",
                "يقيم في {location} منذ سنوات"
            ],
            'ORGANIZATION': [
                "أعلنت {organization} عن نتائجها المالية",
                "وقعت {organization} اتفاقية تعاون",
                "نظمت {organization} ورشة عمل",
                "تبرعت {organization} لصندوق الخير",
                "توسعت {organization} في الأسواق الجديدة"
            ],
            'PHONE': [
                "للاستفسار اتصل على {phone}",
                "رقم الهاتف: {phone}",
                "يمكن التواصل على {phone}",
                "هاتف الطوارئ: {phone}",
                "للحجز: {phone}"
            ]
        }
        
        sentence_id = 0
        
        for category, entries in gazetteers.items():
            if category not in templates:
                continue
            
            # Create sentences using each entry
            for entry in entries[:50]:  # Limit to first 50 for each category
                for template in templates[category]:
                    sentence = template.format(**{category.lower(): entry.text})
                    
                    # Find PII position in sentence
                    pii_start = sentence.find(entry.text)
                    pii_end = pii_start + len(entry.text)
                    
                    training_sentences.append({
                        'sentence_id': f"jordan_gazetteer_{sentence_id}",
                        'text': sentence,
                        'source': 'jordan_gazetteer',
                        'pii_entities': [{
                            'text': entry.text,
                            'type': entry.category,
                            'start': pii_start,
                            'end': pii_end,
                            'confidence': entry.confidence
                        }],
                        'gazetteer_source': entry.source
                    })
                    
                    sentence_id += 1
        
        return training_sentences

    def analyze_gazetteer_coverage(self, gazetteers: Dict[str, List[GazetteerEntry]]):
        """Analyze gazetteer coverage and statistics"""
        print("\n📊 GAZETTEER ANALYSIS")
        print("=" * 40)
        
        total_entries = sum(len(entries) for entries in gazetteers.values())
        print(f"Total gazetteer entries: {total_entries}")
        
        for category, entries in gazetteers.items():
            print(f"\n{category} ({len(entries)} entries):")
            
            # Analyze by subcategory
            subcategories = defaultdict(int)
            sources = defaultdict(int)
            
            for entry in entries:
                subcategories[entry.subcategory] += 1
                sources[entry.source] += 1
            
            print("  Subcategories:")
            for subcat, count in subcategories.items():
                print(f"    {subcat}: {count}")
            
            print("  Sources:")
            for source, count in sources.items():
                print(f"    {source}: {count}")

def main():
    """Main function to create Jordan gazetteers"""
    print("🚀 STARTING JORDAN GAZETTEER CREATION")
    print("=" * 60)
    
    scraper = JordanGazetteerScraper()
    
    # Create all gazetteers
    gazetteers = scraper.create_all_gazetteers()
    
    # Analyze coverage
    scraper.analyze_gazetteer_coverage(gazetteers)
    
    # Save gazetteers
    scraper.save_gazetteers(gazetteers)
    
    # Create training augmentation data
    print("\n🎓 Creating training augmentation data...")
    training_data = scraper.create_training_augmentation_data(gazetteers)
    
    # Save training data
    training_file = "jordan_gazetteers/jordan_training_augmentation.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Created {len(training_data)} training sentences → {training_file}")
    
    print(f"\n🎉 Jordan gazetteer creation completed!")
    print(f"📁 Files saved in jordan_gazetteers/ directory")
    
    return gazetteers, training_data

if __name__ == "__main__":
    gazetteers, training_data = main()
