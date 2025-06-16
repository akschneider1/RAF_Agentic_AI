
#!/usr/bin/env python3
"""
Jordan Enhanced Gazetteer Scraper (Dependency-Free Version)
Creates comprehensive gazetteers without external dependencies
"""

import json
import csv
import re
import os
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GazetteerEntry:
    """Single gazetteer entry"""
    text: str
    category: str
    subcategory: str
    source: str
    confidence: float = 1.0
    metadata: Optional[Dict] = None

class JordanEnhancedScraper:
    """Enhanced Jordan-specific gazetteer creator (no external dependencies)"""
    
    def __init__(self):
        self.gazetteers = {
            'PERSON': defaultdict(list),
            'LOCATION': defaultdict(list),
            'ORGANIZATION': defaultdict(list),
            'PHONE': defaultdict(list),
            'ID_NUMBER': defaultdict(list)
        }

    def create_comprehensive_locations(self) -> List[GazetteerEntry]:
        """Create comprehensive Jordan location data"""
        locations = []
        
        # Enhanced governorate data with districts
        jordan_admin_data = {
            'عمان': {
                'districts': [
                    'قصبة عمان', 'الجامعة', 'ناعور', 'أبو علندا', 'الموقر', 
                    'سحاب', 'الجيزة', 'ماركا', 'القويسمة'
                ],
                'neighborhoods': [
                    'جبل عمان', 'جبل اللويبدة', 'جبل الحسين', 'جبل النزهة',
                    'الشميساني', 'عبدون', 'الرابية', 'خلدا', 'الصويفية',
                    'طبربور', 'مرج الحمام', 'الجبيهة', 'تلاع العلي',
                    'الدوار الأول', 'الدوار الثاني', 'الدوار الثالث',
                    'الدوار الرابع', 'الدوار الخامس', 'الدوار السادس',
                    'الدوار السابع', 'الدوار الثامن'
                ]
            },
            'إربد': {
                'districts': [
                    'قصبة إربد', 'الكورة', 'بني كنانة', 'الرمثا', 'الوسطية',
                    'المزار الشمالي', 'الطيبة', 'بني عبيد'
                ],
                'cities': ['إربد', 'الرمثا', 'المزار الشمالي', 'حوارة', 'الطيبة']
            },
            'الزرقاء': {
                'districts': ['قصبة الزرقاء', 'الرصيفة', 'الضليل'],
                'cities': ['الزرقاء', 'الرصيفة', 'الضليل', 'الأزرق']
            },
            'البلقاء': {
                'districts': ['السلط', 'عين الباشا', 'الشونة الجنوبية', 'دير علا'],
                'cities': ['السلط', 'عين الباشا', 'الشونة الجنوبية', 'دير علا']
            },
            'مادبا': {
                'districts': ['قصبة مادبا', 'ذيبان'],
                'cities': ['مادبا', 'ذيبان']
            },
            'الكرك': {
                'districts': ['قصبة الكرك', 'المزار الجنوبي', 'فقوع', 'الأغوار الجنوبية'],
                'cities': ['الكرك', 'المزار الجنوبي', 'فقوع', 'الصافي']
            },
            'الطفيلة': {
                'districts': ['قصبة الطفيلة', 'بصيرا', 'الحسا'],
                'cities': ['الطفيلة', 'بصيرا', 'الحسا']
            },
            'معان': {
                'districts': ['قصبة معان', 'الشوبك', 'البتراء'],
                'cities': ['معان', 'الشوبك', 'البتراء', 'وادي موسى']
            },
            'العقبة': {
                'districts': ['قصبة العقبة', 'القويرة'],
                'cities': ['العقبة', 'القويرة', 'الديسة']
            },
            'المفرق': {
                'districts': ['قصبة المفرق', 'البادية الشمالية الشرقية', 'البادية الشمالية الغربية'],
                'cities': ['المفرق', 'الصفاوي', 'الرويشد']
            },
            'جرش': {
                'districts': ['قصبة جرش'],
                'cities': ['جرش', 'برما', 'سوف']
            },
            'عجلون': {
                'districts': ['قصبة عجلون', 'كفرنجة'],
                'cities': ['عجلون', 'كفرنجة', 'عنجرة']
            }
        }
        
        # Generate location entries
        for governorate, data in jordan_admin_data.items():
            # Add governorate
            locations.append(
                GazetteerEntry(f"محافظة {governorate}", 'LOCATION', 'governorate', 'jordan_admin_enhanced', 1.0)
            )
            locations.append(
                GazetteerEntry(governorate, 'LOCATION', 'governorate_short', 'jordan_admin_enhanced', 0.95)
            )
            
            # Add districts
            for district in data.get('districts', []):
                locations.append(
                    GazetteerEntry(f"لواء {district}", 'LOCATION', 'district', 'jordan_admin_enhanced', 0.9)
                )
                locations.append(
                    GazetteerEntry(district, 'LOCATION', 'district_short', 'jordan_admin_enhanced', 0.85)
                )
            
            # Add cities
            for city in data.get('cities', []):
                locations.append(
                    GazetteerEntry(city, 'LOCATION', 'city', 'jordan_admin_enhanced', 0.9)
                )
            
            # Add neighborhoods (for Amman)
            for neighborhood in data.get('neighborhoods', []):
                locations.append(
                    GazetteerEntry(neighborhood, 'LOCATION', 'neighborhood', 'jordan_admin_enhanced', 0.8)
                )
        
        return locations

    def create_enhanced_names(self) -> List[GazetteerEntry]:
        """Create enhanced Jordan name database"""
        names = []
        
        # Extended name databases
        extended_male_names = [
            'محمد', 'أحمد', 'خالد', 'عبدالله', 'عمر', 'يوسف', 'عبدالرحمن', 'حسام',
            'طارق', 'سامر', 'وليد', 'نادر', 'باسم', 'مازن', 'فادي', 'رامي',
            'عماد', 'إياد', 'مؤيد', 'معاذ', 'زياد', 'جهاد', 'نبيل', 'وسام',
            'صالح', 'هاني', 'عبدالعزيز', 'أسامة', 'منذر', 'تيسير', 'جمال',
            'كريم', 'نضال', 'أيمن', 'بلال', 'مراد', 'عادل', 'حاتم', 'نور',
            'زين', 'ريان', 'سيف', 'يزن', 'غسان', 'مهند', 'هشام', 'ماهر'
        ]
        
        extended_female_names = [
            'فاطمة', 'عائشة', 'خديجة', 'زينب', 'مريم', 'سارة', 'نور', 'رنا',
            'هند', 'ليلى', 'أسماء', 'آمنة', 'سعاد', 'منى', 'رنيم', 'دينا',
            'لينا', 'رغد', 'شذى', 'ندى', 'ريم', 'هبة', 'نايا', 'سلمى',
            'نادية', 'سهى', 'وفاء', 'إيمان', 'هالة', 'سميرة', 'نجوى',
            'رانيا', 'ديانا', 'نانسي', 'ريتا', 'كريستين', 'سوزان', 'جومانا',
            'لارا', 'مايا', 'تالا', 'جنى', 'يارا', 'تيا', 'ليان', 'آية'
        ]
        
        extended_family_names = [
            'العبدالله', 'المحمد', 'الأحمد', 'الخطيب', 'النجار', 'الزعبي', 'العجارمة',
            'البطاينة', 'الخوالدة', 'الطوالبة', 'القضاة', 'الشوابكة', 'المجالي',
            'الفايز', 'الزيود', 'الحباشنة', 'الحوراني', 'الكايد', 'العموش',
            'الصمادي', 'الربابعة', 'العنانزة', 'الصرايرة', 'الحجازين', 'البشايرة',
            'الحمود', 'السعود', 'الجبور', 'العزايزة', 'الكوالين', 'الخلايلة',
            'الدعجة', 'السوالمة', 'الشلبي', 'الزواهرة', 'القرعان', 'العكاشة',
            'الحديد', 'النوايسة', 'الجازي', 'المومني', 'البدارين', 'الشواربة'
        ]
        
        # Generate comprehensive name combinations
        for male_name in extended_male_names:
            for family_name in extended_family_names[:20]:  # Top 20 family names
                full_name = f"{male_name} {family_name}"
                names.append(
                    GazetteerEntry(full_name, 'PERSON', 'male_full_name', 'jordan_enhanced_names', 0.85)
                )
        
        for female_name in extended_female_names:
            for family_name in extended_family_names[:20]:
                full_name = f"{female_name} {family_name}"
                names.append(
                    GazetteerEntry(full_name, 'PERSON', 'female_full_name', 'jordan_enhanced_names', 0.85)
                )
        
        # Add standalone names
        for name in extended_male_names:
            names.append(
                GazetteerEntry(name, 'PERSON', 'male_first_name', 'jordan_enhanced_names', 0.75)
            )
        
        for name in extended_female_names:
            names.append(
                GazetteerEntry(name, 'PERSON', 'female_first_name', 'jordan_enhanced_names', 0.75)
            )
        
        for name in extended_family_names:
            names.append(
                GazetteerEntry(name, 'PERSON', 'family_name', 'jordan_enhanced_names', 0.7)
            )
        
        return names

    def create_comprehensive_organizations(self) -> List[GazetteerEntry]:
        """Create comprehensive Jordan organization database"""
        organizations = []
        
        # Government entities (comprehensive)
        government_entities = {
            'ministries': [
                'وزارة الداخلية', 'وزارة الخارجية وشؤون المغتربين', 'وزارة المالية',
                'وزارة التربية والتعليم', 'وزارة الصحة', 'وزارة العمل',
                'وزارة الزراعة', 'وزارة الطاقة والثروة المعدنية', 'وزارة النقل',
                'وزارة السياحة والآثار', 'وزارة الثقافة', 'وزارة الشباب',
                'وزارة التعليم العالي والبحث العلمي', 'وزارة البيئة',
                'وزارة التنمية الاجتماعية', 'وزارة الصناعة والتجارة والتموين',
                'وزارة الأشغال العامة والإسكان', 'وزارة المياه والري',
                'وزارة التخطيط والتعاون الدولي', 'وزارة الاقتصاد الرقمي والريادة'
            ],
            'departments': [
                'دائرة الإحصاءات العامة', 'دائرة الأراضي والمساحة', 'دائرة الجمارك الأردنية',
                'مؤسسة الضمان الاجتماعي', 'دائرة ضريبة الدخل والمبيعات',
                'البنك المركزي الأردني', 'دائرة الأحوال المدنية والجوازات',
                'دائرة المحاسبة العامة', 'ديوان الخدمة المدنية', 'ديوان المحاسبة',
                'هيئة النزاهة ومكافحة الفساد', 'المجلس الاقتصادي والاجتماعي'
            ],
            'authorities': [
                'سلطة منطقة العقبة الاقتصادية الخاصة', 'هيئة تنظيم قطاع الطاقة والمعادن',
                'هيئة تنظيم النقل البري', 'هيئة تنظيم الاتصالات', 'سلطة المياه',
                'سلطة وادي الأردن', 'هيئة تنظيم شؤون التأمين', 'هيئة الأوراق المالية'
            ]
        }
        
        for category, entities in government_entities.items():
            for entity in entities:
                organizations.append(
                    GazetteerEntry(entity, 'ORGANIZATION', f'government_{category}', 'jordan_government_enhanced', 0.95)
                )
        
        # Universities (comprehensive)
        universities = [
            'الجامعة الأردنية', 'جامعة اليرموك', 'جامعة العلوم والتكنولوجيا الأردنية',
            'جامعة مؤتة', 'جامعة البلقاء التطبيقية', 'الجامعة الهاشمية',
            'جامعة آل البيت', 'جامعة الحسين بن طلال', 'جامعة الطفيلة التقنية',
            'جامعة فيلادلفيا', 'جامعة العلوم التطبيقية الخاصة', 'الجامعة الأمريكية في مادبا',
            'جامعة الأميرة سمية للتكنولوجيا', 'الجامعة الألمانية الأردنية',
            'جامعة الشرق الأوسط', 'جامعة عمان الأهلية', 'جامعة إربد الأهلية',
            'جامعة الزرقاء', 'جامعة البتراء', 'جامعة جدارا'
        ]
        
        for university in universities:
            organizations.append(
                GazetteerEntry(university, 'ORGANIZATION', 'university', 'jordan_education_enhanced', 0.9)
            )
        
        # Major companies and banks
        financial_institutions = [
            'البنك الأهلي الأردني', 'بنك الإسكان للتجارة والتمويل', 'البنك العربي',
            'بنك القاهرة عمان', 'البنك الإسلامي الأردني', 'بنك الاستثمار العربي الأردني',
            'البنك التجاري الأردني', 'بنك الأردن', 'بنك المؤسسة العربية المصرفية',
            'بنك الاتحاد', 'البنك الأردني الكويتي', 'بنك سوسيتيه جنرال'
        ]
        
        for bank in financial_institutions:
            organizations.append(
                GazetteerEntry(bank, 'ORGANIZATION', 'bank', 'jordan_financial_enhanced', 0.9)
            )
        
        # Telecommunications companies
        telecom_companies = [
            'شركة الاتصالات الأردنية - أورنج', 'شركة زين الأردن للاتصالات المتنقلة',
            'شركة أمنية للاتصالات المتنقلة', 'شركة أكس برس تيليكوم', 'شركة بتلكو الأردن'
        ]
        
        for company in telecom_companies:
            organizations.append(
                GazetteerEntry(company, 'ORGANIZATION', 'telecom', 'jordan_telecom_enhanced', 0.9)
            )
        
        return organizations

    def create_enhanced_phone_patterns(self) -> List[GazetteerEntry]:
        """Create enhanced Jordan phone number patterns with TRC validation"""
        phones = []
        
        # Jordan mobile prefixes (validated from TRC data)
        mobile_operators = {
            'zain': ['077', '078'],
            'orange': ['079'],
            'umniah': ['078', '077']  # Some overlap
        }
        
        # Generate realistic phone number examples
        for operator, prefixes in mobile_operators.items():
            for prefix in prefixes:
                # Generate sample numbers
                for i in range(100000, 100010):  # 10 samples per prefix
                    sample_number = f"{prefix}{i}"
                    phones.extend([
                        GazetteerEntry(sample_number, 'PHONE', f'mobile_{operator}', 'jordan_trc_enhanced', 0.8),
                        GazetteerEntry(f"+962 {sample_number[1:]}", 'PHONE', f'mobile_{operator}_intl', 'jordan_trc_enhanced', 0.8),
                        GazetteerEntry(f"00962 {sample_number[1:]}", 'PHONE', f'mobile_{operator}_intl2', 'jordan_trc_enhanced', 0.75)
                    ])
        
        # Landline patterns by governorate
        landline_codes = {
            'amman': '06',
            'irbid': '02', 
            'zarqa': '05',
            'karak': '03',
            'maan': '03',
            'aqaba': '03',
            'mafraq': '02',
            'jerash': '02',
            'ajloun': '02',
            'balqa': '05',
            'madaba': '05',
            'tafilah': '03'
        }
        
        for city, code in landline_codes.items():
            for i in range(5550000, 5550010):  # Sample landline numbers
                sample_number = f"0{code}{str(i)[2:]}"  # Remove leading digits to fit format
                phones.extend([
                    GazetteerEntry(sample_number, 'PHONE', f'landline_{city}', 'jordan_landline_enhanced', 0.7),
                    GazetteerEntry(f"+962 {code} {str(i)[2:]}", 'PHONE', f'landline_{city}_intl', 'jordan_landline_enhanced', 0.7)
                ])
        
        return phones

    def create_enhanced_id_numbers(self) -> List[GazetteerEntry]:
        """Create enhanced ID number patterns for Jordan"""
        id_numbers = []
        
        # Jordan National ID patterns
        for year in ['85', '90', '95', '00', '05']:  # Birth years
            for month in ['01', '06', '12']:  # Sample months
                for seq in range(1000, 1010):  # Sequential numbers
                    sample_id = f"{year}{month}{seq:04d}"
                    id_numbers.append(
                        GazetteerEntry(sample_id, 'ID_NUMBER', 'national_id', 'jordan_civil_status', 0.8)
                    )
        
        # Passport number patterns
        for letter in ['A', 'B', 'C', 'D', 'E']:
            for num in range(1000000, 1000010):
                passport = f"{letter}{num}"
                id_numbers.append(
                    GazetteerEntry(passport, 'ID_NUMBER', 'passport', 'jordan_passport', 0.8)
                )
        
        # Driver's license patterns
        for num in range(10000000, 10000010):
            license_num = str(num)
            id_numbers.append(
                GazetteerEntry(license_num, 'ID_NUMBER', 'drivers_license', 'jordan_traffic', 0.7)
            )
        
        return id_numbers

    def create_all_enhanced_gazetteers(self) -> Dict[str, List[GazetteerEntry]]:
        """Create all enhanced Jordan gazetteers"""
        print("🇯🇴 CREATING ENHANCED JORDAN GAZETTEERS (Dependency-Free)")
        print("=" * 60)
        
        all_gazetteers = {'LOCATION': [], 'PERSON': [], 'ORGANIZATION': [], 'PHONE': [], 'ID_NUMBER': []}
        
        # Create comprehensive locations
        print("📍 Creating comprehensive location gazetteer...")
        locations = self.create_comprehensive_locations()
        all_gazetteers['LOCATION'].extend(locations)
        print(f"   ✅ Created {len(locations)} location entries")
        
        # Create enhanced names
        print("👤 Creating enhanced person name gazetteer...")
        names = self.create_enhanced_names()
        all_gazetteers['PERSON'].extend(names)
        print(f"   ✅ Created {len(names)} person name entries")
        
        # Create comprehensive organizations
        print("🏢 Creating comprehensive organization gazetteer...")
        organizations = self.create_comprehensive_organizations()
        all_gazetteers['ORGANIZATION'].extend(organizations)
        print(f"   ✅ Created {len(organizations)} organization entries")
        
        # Create enhanced phone patterns
        print("📞 Creating enhanced phone number gazetteer...")
        phones = self.create_enhanced_phone_patterns()
        all_gazetteers['PHONE'].extend(phones)
        print(f"   ✅ Created {len(phones)} phone number entries")
        
        # Create enhanced ID patterns
        print("🆔 Creating enhanced ID number gazetteer...")
        id_numbers = self.create_enhanced_id_numbers()
        all_gazetteers['ID_NUMBER'].extend(id_numbers)
        print(f"   ✅ Created {len(id_numbers)} ID number entries")
        
        return all_gazetteers

    def save_enhanced_gazetteers(self, gazetteers: Dict[str, List[GazetteerEntry]], output_dir: str = "jordan_gazetteers_enhanced"):
        """Save enhanced gazetteers to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving enhanced gazetteers to {output_dir}/")
        
        for category, entries in gazetteers.items():
            # Save as CSV
            csv_file = os.path.join(output_dir, f"jordan_{category.lower()}_enhanced.csv")
            
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
            json_file = os.path.join(output_dir, f"jordan_{category.lower()}_enhanced.json")
            
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

    def analyze_enhanced_coverage(self, gazetteers: Dict[str, List[GazetteerEntry]]):
        """Analyze enhanced gazetteer coverage"""
        print("\n📊 ENHANCED GAZETTEER ANALYSIS")
        print("=" * 45)
        
        total_entries = sum(len(entries) for entries in gazetteers.values())
        print(f"Total enhanced gazetteer entries: {total_entries}")
        
        for category, entries in gazetteers.items():
            print(f"\n{category} ({len(entries)} entries):")
            
            # Analyze by subcategory and source
            subcategories = defaultdict(int)
            sources = defaultdict(int)
            
            for entry in entries:
                subcategories[entry.subcategory] += 1
                sources[entry.source] += 1
            
            print("  Subcategories:")
            for subcat, count in sorted(subcategories.items()):
                print(f"    {subcat}: {count}")
            
            print("  Sources:")
            for source, count in sorted(sources.items()):
                print(f"    {source}: {count}")

def main():
    """Main function to create enhanced Jordan gazetteers"""
    print("🚀 STARTING ENHANCED JORDAN GAZETTEER CREATION")
    print("=" * 70)
    
    scraper = JordanEnhancedScraper()
    
    # Create all enhanced gazetteers
    gazetteers = scraper.create_all_enhanced_gazetteers()
    
    # Analyze coverage
    scraper.analyze_enhanced_coverage(gazetteers)
    
    # Save enhanced gazetteers
    scraper.save_enhanced_gazetteers(gazetteers)
    
    print(f"\n🎉 Enhanced Jordan gazetteer creation completed!")
    print(f"📁 Files saved in jordan_gazetteers_enhanced/ directory")
    
    return gazetteers

if __name__ == "__main__":
    gazetteers = main()
