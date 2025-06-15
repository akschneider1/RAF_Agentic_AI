
import random
import re
from typing import List, Dict
from dataclasses import dataclass
from rules import PIIDetector

@dataclass
class SyntheticTemplate:
    """Template for generating synthetic sentences"""
    template: str
    pii_types: List[str]
    category: str

class SyntheticPIIGenerator:
    """Generates synthetic Arabic sentences with PII"""
    
    def __init__(self):
        self.detector = PIIDetector()
        self.templates = self._create_templates()
        self.gazetteers = self._create_gazetteers()
    
    def _create_templates(self) -> List[SyntheticTemplate]:
        """Create diverse Arabic sentence templates"""
        templates = []
        
        # NAME templates (20 diverse templates)
        name_templates = [
            "تم تعيين {name} في منصب جديد.",
            "التقيت مع {name} في المكتب اليوم.",
            "قال الدكتور {name} أن النتائج ممتازة.",
            "تم ترقية {name} إلى منصب مدير عام.",
            "حضر الأستاذ {name} الاجتماع الصباحي.",
            "اتصل {name} لتأكيد الموعد غداً.",
            "أرسل {name} التقرير عبر البريد الإلكتروني.",
            "المهندس {name} مسؤول عن هذا المشروع.",
            "تخرج {name} من الجامعة بتقدير ممتاز.",
            "عين {name} رئيساً للقسم الجديد.",
            "استقبل {name} الوفد الزائر بحفاوة.",
            "قدم {name} عرضاً رائعاً أمس.",
            "سافر {name} إلى دبي لحضور المؤتمر.",
            "نال {name} جائزة أفضل موظف.",
            "أدار {name} الجلسة بكفاءة عالية.",
            "شارك {name} في المناقشات المهمة.",
            "وافق {name} على الاقتراح المقدم.",
            "درس {name} الملف بعناية فائقة.",
            "اقترح {name} حلولاً إبداعية للمشكلة.",
            "مثل {name} الشركة في المعرض."
        ]
        
        # ADDRESS templates (20 diverse templates)
        address_templates = [
            "العنوان الجديد هو {address}.",
            "انتقلت الشركة إلى {address}.",
            "يقع المكتب في {address}.",
            "تم تسليم الطلب إلى {address}.",
            "العيادة موجودة في {address}.",
            "سيقام الحفل في {address}.",
            "المطعم الجديد يقع في {address}.",
            "تم افتتاح الفرع في {address}.",
            "توجه إلى {address} للقاء العميل.",
            "أرسل الدعوة إلى {address}.",
            "المقر الرئيسي في {address}.",
            "البناء الجديد في {address} جاهز.",
            "انعقدت الدورة في {address}.",
            "تقع المدرسة في {address}.",
            "المستشفى الجديد في {address}.",
            "تم تغيير العنوان إلى {address}.",
            "الفندق يقع في {address}.",
            "المركز التجاري في {address}.",
            "المصنع الجديد في {address}.",
            "تم نقل المكتب إلى {address}."
        ]
        
        # ID_NUMBER templates (20 diverse templates)
        id_templates = [
            "رقم الهوية الوطنية: {id_number}.",
            "الرقم المدني للعميل هو {id_number}.",
            "تم تسجيل الهوية رقم {id_number}.",
            "يحمل رقم هوية {id_number}.",
            "بطاقة الهوية رقم {id_number} منتهية الصلاحية.",
            "تم تحديث بيانات الهوية {id_number}.",
            "رقم السجل المدني: {id_number}.",
            "الهوية الشخصية رقم {id_number}.",
            "يرجى تقديم الهوية رقم {id_number}.",
            "تم إصدار هوية جديدة برقم {id_number}.",
            "رقم البطاقة الشخصية {id_number}.",
            "تحقق من صحة الرقم {id_number}.",
            "الهوية المدنية {id_number} صالحة.",
            "سجل برقم الهوية {id_number}.",
            "البطاقة رقم {id_number} تحتاج تجديد.",
            "رقم الوثيقة الشخصية: {id_number}.",
            "تم استلام الهوية {id_number}.",
            "الرقم الثابت للهوية: {id_number}.",
            "بطاقة التعريف رقم {id_number}.",
            "رقم الإثبات الشخصي: {id_number}."
        ]
        
        # PHONE templates (15 templates)
        phone_templates = [
            "رقم الهاتف: {phone}.",
            "اتصل على {phone} للاستفسار.",
            "رقم الجوال الجديد: {phone}.",
            "تواصل معنا على {phone}.",
            "للحجز اتصل على {phone}.",
            "رقم الطوارئ: {phone}.",
            "هاتف العمل: {phone}.",
            "يمكن الوصول إليه على {phone}.",
            "رقم الهاتف المحمول: {phone}.",
            "للمزيد من المعلومات: {phone}.",
            "خط ساخن: {phone}.",
            "رقم الاتصال المباشر: {phone}.",
            "يرجى الاتصال على {phone}.",
            "هاتف خدمة العملاء: {phone}.",
            "للشكاوى والاقتراحات: {phone}."
        ]
        
        # EMAIL templates (15 templates)
        email_templates = [
            "البريد الإلكتروني: {email}.",
            "راسلنا على {email}.",
            "للاستفسارات: {email}.",
            "أرسل إلى {email}.",
            "البريد الرسمي: {email}.",
            "للدعم الفني: {email}.",
            "ايميل الشركة: {email}.",
            "تواصل عبر {email}.",
            "البريد المؤسسي: {email}.",
            "للمراسلات: {email}.",
            "عنوان الإيميل: {email}.",
            "البريد الشخصي: {email}.",
            "للتواصل: {email}.",
            "ايميل المدير: {email}.",
            "البريد الخاص: {email}."
        ]
        
        # Convert to SyntheticTemplate objects
        for template in name_templates:
            templates.append(SyntheticTemplate(template, ['PERSON'], 'NAME'))
        
        for template in address_templates:
            templates.append(SyntheticTemplate(template, ['ADDRESS'], 'ADDRESS'))
        
        for template in id_templates:
            templates.append(SyntheticTemplate(template, ['ID_NUMBER'], 'ID'))
        
        for template in phone_templates:
            templates.append(SyntheticTemplate(template, ['PHONE'], 'PHONE'))
        
        for template in email_templates:
            templates.append(SyntheticTemplate(template, ['EMAIL'], 'EMAIL'))
        
        # Multi-PII templates
        multi_templates = [
            "السيد {name} يسكن في {address} ورقم هاتفه {phone}.",
            "تواصل مع {name} على {email} أو زره في {address}.",
            "هوية {name} رقم {id_number} والعنوان {address}.",
            "اتصل بـ {name} على {phone} أو راسله على {email}.",
            "بيانات العميل: {name}، هاتف: {phone}، هوية: {id_number}.",
            "المدير {name} مكتبه في {address} وإيميله {email}.",
            "سجل {name} عنوانه {address} ورقم {phone}.",
        ]
        
        for template in multi_templates:
            pii_types = []
            if '{name}' in template:
                pii_types.append('PERSON')
            if '{address}' in template:
                pii_types.append('ADDRESS')
            if '{phone}' in template:
                pii_types.append('PHONE')
            if '{email}' in template:
                pii_types.append('EMAIL')
            if '{id_number}' in template:
                pii_types.append('ID_NUMBER')
            
            templates.append(SyntheticTemplate(template, pii_types, 'MULTI'))
        
        return templates
    
    def _create_gazetteers(self) -> Dict[str, List[str]]:
        """Create gazetteers for different PII types"""
        return {
            'PERSON': [
                # Arabic names
                'أحمد محمد علي', 'فاطمة أحمد', 'محمد عبدالله', 'عائشة سالم',
                'خالد عبدالعزيز', 'نورا أحمد', 'سعد الدين', 'هند محمود',
                'عبدالرحمن يوسف', 'مريم علي', 'طارق السيد', 'زينب حسن',
                'عمر الفاروق', 'ليلى عبدالله', 'يوسف أحمد', 'سارة محمد',
                'حسام الدين', 'رقية عبدالرحمن', 'معاذ بن جبل', 'آمنة عثمان',
                'إبراهيم خليل', 'خديجة أحمد', 'عثمان بن عفان', 'عبير سعد',
                'حمزة محمود', 'أسماء يوسف', 'عبدالله محمد', 'منى عبدالعزيز',
                'صالح أحمد', 'هاجر إبراهيم', 'محمود عبدالله', 'ريم خالد'
            ],
            
            'ADDRESS': [
                # Saudi addresses
                'طريق الملك فهد، الرياض', 'شارع التحلية، جدة', 'طريق الأمير محمد بن فهد، الدمام',
                'حي النزهة، الرياض', 'شارع الأمير سلطان، جدة', 'طريق الملك عبدالعزيز، مكة',
                'حي المروج، الرياض', 'شارع المدينة المنورة، جدة', 'طريق الدائري الشرقي، الرياض',
                'حي الروضة، جدة', 'شارع الملك خالد، أبها', 'طريق الملك فيصل، تبوك',
                'حي السليمانية، الرياض', 'شارع حراء، مكة', 'طريق الأمير تركي، الرياض',
                
                # UAE addresses  
                'شارع الشيخ زايد، دبي', 'طريق الكورنيش، أبوظبي', 'شارع المطار، دبي',
                'منطقة دبي مارينا', 'حي البرشاء، دبي', 'منطقة الزعفرانة، أبوظبي',
                
                # Jordan addresses
                'شارع الملكة رانيا، عمان', 'دوار الداخلية، عمان', 'شارع المدينة المنورة، إربد',
                
                # Egypt addresses  
                'شارع التحرير، القاهرة', 'كورنيش النيل، الأقصر', 'شارع الهرم، الجيزة'
            ],
            
            'ID_NUMBER': [
                # Saudi National IDs
                '1234567890', '2123456789', '1987654321', '2876543210',
                '1456789012', '2345678901', '1789012345', '2012345678',
                '1567890123', '2678901234', '1890123456', '2234567890',
                
                # UAE Emirates IDs
                '784-2020-1234567-1', '784-2019-9876543-2', '784-2021-5555555-3',
                '784-2018-7777777-4', '784-2022-3333333-5', '784-2020-9999999-6',
                
                # Egyptian National IDs
                '29801011234567', '28505051234567', '29912121234567',
                '28707071234567', '29606061234567', '28404041234567'
            ],
            
            'PHONE': [
                # Saudi mobile numbers
                '+966 50 123 4567', '+966 55 987 6543', '+966 56 444 5555',
                '0501234567', '0559876543', '0564445555', '0571112222',
                '+966 58 777 8888', '0507777777', '+966 59 333 4444',
                
                # UAE numbers
                '+971 50 123 4567', '+971 55 987 6543', '+971 56 444 5555',
                
                # Jordan numbers  
                '+962 77 123 4567', '+962 79 987 6543', '077 123 4567',
                
                # Egypt numbers
                '+20 10 1234 5678', '+20 11 9876 5432', '+20 12 4444 5555'
            ],
            
            'EMAIL': [
                'ahmed@gmail.com', 'fatima@hotmail.com', 'mohamed@outlook.com',
                'sara@yahoo.com', 'khaled@company.sa', 'nora@university.edu.sa',
                'hassan@bank.com', 'layla@hospital.org', 'omar@tech.ae',
                'maryam@consulting.com', 'youssef@trading.jo', 'zainab@clinic.eg',
                'ibrahim@school.edu', 'asma@government.gov.sa', 'saleh@business.com',
                'hind@medical.org', 'abdullah@finance.sa', 'amina@legal.ae'
            ]
        }
    
    def generate_pii_value(self, pii_type: str) -> str:
        """Generate a random PII value of the specified type"""
        if pii_type in self.gazetteers:
            return random.choice(self.gazetteers[pii_type])
        
        # Fallback generation
        if pii_type == 'PERSON':
            return 'أحمد محمد'
        elif pii_type == 'ADDRESS':
            return 'شارع الملك فهد، الرياض'
        elif pii_type == 'ID_NUMBER':
            return f"{random.randint(1, 2)}{random.randint(100000000, 999999999)}"
        elif pii_type == 'PHONE':
            return f"+966 5{random.randint(0, 9)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
        elif pii_type == 'EMAIL':
            return f"user{random.randint(1, 1000)}@example.com"
        else:
            return f"[{pii_type}]"
    
    def generate_sentence(self, template: SyntheticTemplate) -> Tuple[str, Dict]:
        """Generate a sentence from a template with PII values"""
        sentence = template.template
        pii_info = {}
        
        # Replace placeholders with actual PII
        for pii_type in template.pii_types:
            placeholder_map = {
                'PERSON': 'name',
                'ADDRESS': 'address', 
                'ID_NUMBER': 'id_number',
                'PHONE': 'phone',
                'EMAIL': 'email'
            }
            
            placeholder = placeholder_map.get(pii_type, pii_type.lower())
            if f'{{{placeholder}}}' in sentence:
                pii_value = self.generate_pii_value(pii_type)
                sentence = sentence.replace(f'{{{placeholder}}}', pii_value)
                pii_info[pii_type] = pii_value
        
        return sentence, pii_info
    
    def generate_dataset(self, num_sentences: int = 10000) -> List[Dict]:
        """Generate a synthetic dataset"""
        dataset = []
        
        for i in range(num_sentences):
            # Choose random template
            template = random.choice(self.templates)
            
            # Generate sentence
            sentence, pii_info = self.generate_sentence(template)
            
            # Detect PII using our rules engine
            detected_pii = self.detector.detect_all_pii(sentence, min_confidence=0.7)
            
            dataset.append({
                'sentence_id': i,
                'text': sentence,
                'template_category': template.category,
                'expected_pii': pii_info,
                'detected_pii': [
                    {
                        'text': match.text,
                        'type': match.pii_type,
                        'start': match.start_pos,
                        'end': match.end_pos,
                        'confidence': match.confidence
                    }
                    for match in detected_pii
                ]
            })
        
        return dataset

def test_synthetic_generator():
    """Test the synthetic generator"""
    generator = SyntheticPIIGenerator()
    
    print("SYNTHETIC PII GENERATOR TEST")
    print("=" * 50)
    
    # Test individual templates
    print("Sample generated sentences:")
    print("-" * 30)
    
    for i in range(10):
        template = random.choice(generator.templates)
        sentence, pii_info = generator.generate_sentence(template)
        print(f"{i+1}. {sentence}")
        print(f"   PII: {pii_info}")
        print()

if __name__ == "__main__":
    test_synthetic_generator()
