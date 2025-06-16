
#!/usr/bin/env python3
"""
Runner script for Jordan Enhanced Scraper
Handles dependencies and error recovery
"""

import sys
import os

def check_dependencies():
    """Check if required modules are available"""
    missing = []
    
    try:
        import requests
    except ImportError:
        missing.append('requests')
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Installing missing packages...")
        
        for package in missing:
            os.system(f"pip install {package}")
    
    return len(missing) == 0

def run_enhanced_scraper():
    """Run the enhanced scraper with error handling"""
    try:
        from jordan_enhanced_scraper import JordanEnhancedScraper, main
        
        print("🚀 Starting Enhanced Jordan Scraper...")
        
        # Run the scraper
        enhanced_data = main()
        
        print("✅ Enhanced scraper completed successfully!")
        return enhanced_data
        
    except Exception as e:
        print(f"❌ Error running enhanced scraper: {e}")
        print("\nFalling back to basic gazetteer scraper...")
        
        try:
            from jordan_gazetteer_scraper import main as basic_main
            return basic_main()
        except Exception as e2:
            print(f"❌ Basic scraper also failed: {e2}")
            return None

if __name__ == "__main__":
    # Check dependencies first
    if check_dependencies():
        enhanced_data = run_enhanced_scraper()
        
        if enhanced_data:
            total_entries = sum(len(entries) for entries in enhanced_data.values() if hasattr(entries, '__len__'))
            print(f"\n🎉 Successfully created {total_entries} enhanced gazetteer entries!")
        else:
            print("\n⚠️ Enhanced scraper failed. Check the basic gazetteer scraper instead.")
    else:
        print("❌ Could not install required dependencies.")
