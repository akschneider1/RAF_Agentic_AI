
#!/usr/bin/env python3
"""
Jordan Gazetteer Setup Script
Main script to create Jordan gazetteers and integrate them with the PII detection system
"""

import os
import sys
from jordan_gazetteer_scraper import JordanGazetteerScraper, main as create_gazetteers
from gazetteer_integration import GazetteerEnhancedDetector, test_gazetteer_integration

def setup_jordan_gazetteers():
    """Complete setup of Jordan gazetteers"""
    print("🇯🇴 JORDAN GAZETTEER SETUP")
    print("=" * 60)
    
    # Step 1: Create gazetteers
    print("📚 Step 1: Creating Jordan gazetteers...")
    gazetteers, training_data = create_gazetteers()
    
    # Step 2: Test integration
    print("\n🔧 Step 2: Testing gazetteer integration...")
    test_gazetteer_integration()
    
    # Step 3: Update synthetic generator to use gazetteers
    print("\n🔄 Step 3: Updating synthetic data generator...")
    update_synthetic_generator_with_gazetteers()
    
    # Step 4: Create enhanced training dataset
    print("\n📈 Step 4: Creating enhanced training dataset...")
    create_enhanced_training_data()
    
    print("\n✅ Jordan gazetteer setup completed successfully!")
    
    return True

def update_synthetic_generator_with_gazetteers():
    """Update the synthetic generator to use Jordan gazetteers"""
    try:
        from synthetic_generator import SyntheticPIIGenerator
        import json
        
        # Load Jordan gazetteers
        gazetteer_files = {
            'PERSON': 'jordan_gazetteers/jordan_person.json',
            'LOCATION': 'jordan_gazetteers/jordan_location.json',
            'ORGANIZATION': 'jordan_gazetteers/jordan_organization.json',
            'PHONE': 'jordan_gazetteers/jordan_phone.json'
        }
        
        enhanced_gazetteers = {}
        
        for category, filepath in gazetteer_files.items():
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract just the text values for synthetic generator
                enhanced_gazetteers[category] = [entry['text'] for entry in data]
                print(f"  ✅ Loaded {len(enhanced_gazetteers[category])} {category} entries")
        
        # Save enhanced gazetteers for synthetic generator
        output_file = 'enhanced_gazetteers.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_gazetteers, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 Enhanced gazetteers saved to {output_file}")
        
    except Exception as e:
        print(f"  ❌ Error updating synthetic generator: {e}")

def create_enhanced_training_data():
    """Create training data that combines Wojood + synthetic + gazetteer data"""
    try:
        from data_augmentation import DataAugmentation
        import json
        import pandas as pd
        
        # Load gazetteer training data
        gazetteer_training_file = 'jordan_gazetteers/jordan_training_augmentation.json'
        
        if os.path.exists(gazetteer_training_file):
            with open(gazetteer_training_file, 'r', encoding='utf-8') as f:
                gazetteer_data = json.load(f)
            
            print(f"  📊 Loaded {len(gazetteer_data)} gazetteer training sentences")
            
            # Convert to DataFrame format compatible with existing augmentation
            gazetteer_df = pd.DataFrame(gazetteer_data)
            
            # Save as enhanced training data
            enhanced_file = 'train_enhanced_with_gazetteers.json'
            gazetteer_df.to_json(enhanced_file, orient='records', force_ascii=False, indent=2)
            
            print(f"  💾 Enhanced training data saved to {enhanced_file}")
            
        else:
            print(f"  ⚠️  Gazetteer training file not found: {gazetteer_training_file}")
    
    except Exception as e:
        print(f"  ❌ Error creating enhanced training data: {e}")

def demonstrate_enhanced_detection():
    """Demonstrate the enhanced PII detection with gazetteers"""
    print("\n🎯 DEMONSTRATING ENHANCED PII DETECTION")
    print("=" * 50)
    
    # Test cases specifically for Jordan
    jordan_test_cases = [
        "السيد عبدالله المجالي يعمل مديراً في البنك الأهلي الأردني في عمان",
        "الدكتور أحمد الزعبي من جامعة اليرموك في محافظة إربد",
        "للتواصل مع المهندس خالد العموش اتصل على 077123456",
        "تقع شركة زين الأردن في منطقة الشميساني في عمان",
        "سافرت فاطمة الطوالبة إلى مدينة العقبة لحضور المؤتمر",
        "وزارة التربية والتعليم أعلنت عن نتائج الثانوية العامة",
        "رقم هاتف المستشفى الأردني: +962 6 5551234"
    ]
    
    try:
        detector = GazetteerEnhancedDetector()
        
        for i, text in enumerate(jordan_test_cases, 1):
            print(f"\n📝 Test {i}:")
            print(f"Text: {text}")
            print("Results:")
            
            matches = detector.combine_all_matches(text, min_confidence=0.6)
            
            if matches:
                for match in matches:
                    source_info = f"({match['source']}"
                    if match['source'] == 'gazetteer':
                        source_info += f" - {match.get('gazetteer_source', 'unknown')}"
                    source_info += ")"
                    
                    print(f"  • {match['type']}: '{match['text']}' "
                          f"[{match['confidence']:.2f}] {source_info}")
            else:
                print("  No PII detected")
    
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")

def main():
    """Main execution function"""
    print("🚀 STARTING JORDAN GAZETTEER SETUP")
    print("=" * 60)
    
    try:
        # Complete setup
        setup_jordan_gazetteers()
        
        # Demonstrate capabilities
        demonstrate_enhanced_detection()
        
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("📁 Generated files:")
        print("  • jordan_gazetteers/ - Gazetteer data files")
        print("  • enhanced_gazetteers.json - For synthetic generator")
        print("  • train_enhanced_with_gazetteers.json - Enhanced training data")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
