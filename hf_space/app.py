
import gradio as gr
import sys
import os

# Add the parent directory to the path to import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rules import PIIDetector
    detector = PIIDetector()
    detector_available = True
except ImportError:
    detector_available = False
    print("Warning: PIIDetector not available. Using fallback.")

def detect_arabic_pii(text, confidence_threshold=0.7, use_obfuscation=False):
    """Detect and mask PII in Arabic/English text"""
    if not text.strip():
        return "", "No text provided", ""
    
    if not detector_available:
        return text, "PII Detection engine not available", "Please check the setup"
    
    try:
        # Detect PII
        detected_pii = detector.detect_all_pii(text, confidence_threshold)
        
        # Simple masking
        masked_text = text
        for match in sorted(detected_pii, key=lambda x: x.start_pos, reverse=True):
            if use_obfuscation:
                # Use consistent obfuscation
                surrogate = _get_surrogate(match.pii_type, match.text)
                masked_text = masked_text[:match.start_pos] + surrogate + masked_text[match.end_pos:]
            else:
                masked_text = masked_text[:match.start_pos] + "[MASKED]" + masked_text[match.end_pos:]
        
        # Create summary
        summary_counts = {}
        for match in detected_pii:
            summary_counts[match.pii_type] = summary_counts.get(match.pii_type, 0) + 1
        
        summary = f"Detected {len(detected_pii)} PII entities:\n"
        for pii_type, count in summary_counts.items():
            summary += f"• {pii_type}: {count}\n"
        
        # Create details
        details = "Detected PII Details:\n"
        for i, match in enumerate(detected_pii, 1):
            details += f"{i}. {match.pii_type}: '{match.text}' (confidence: {match.confidence:.2f})\n"
        
        return masked_text, summary, details
        
    except Exception as e:
        return text, f"Error: {str(e)}", "Please try again"

def _get_surrogate(pii_type, original_text):
    """Generate consistent surrogate values"""
    import hashlib
    
    surrogate_map = {
        "PHONE": ["05XXXXXXXX", "01XXXXXXXX", "02XXXXXXXX"],
        "EMAIL": ["user@example.com", "contact@domain.com"],
        "PERSON": ["أحمد المجهول", "سارة المجهولة", "محمد المخفي"],
        "LOCATION": ["المدينة الأولى", "المدينة الثانية"],
        "ID_NUMBER": ["1XXXXXXXXX", "2XXXXXXXXX"],
        "ORGANIZATION": ["الشركة الأولى", "المؤسسة الثانية"]
    }
    
    surrogates = surrogate_map.get(pii_type, ["[MASKED]"])
    hash_val = int(hashlib.md5(f"{pii_type}:{original_text}".encode()).hexdigest()[:8], 16)
    return surrogates[hash_val % len(surrogates)]

# Create Gradio interface
demo = gr.Interface(
    fn=detect_arabic_pii,
    inputs=[
        gr.Textbox(
            label="Arabic/English Text", 
            placeholder="Enter text containing potential PII...",
            lines=5
        ),
        gr.Slider(
            minimum=0.1, 
            maximum=1.0, 
            value=0.7, 
            label="Confidence Threshold"
        ),
        gr.Checkbox(
            label="Use Consistent Obfuscation", 
            value=False
        )
    ],
    outputs=[
        gr.Textbox(label="Masked Text", lines=5),
        gr.Textbox(label="Summary", lines=3),
        gr.Textbox(label="Details", lines=10)
    ],
    title="🔍 Arabic PII Detection System",
    description="Advanced Arabic and English PII detection with hybrid ML+Rules approach",
    examples=[
        ["اتصل بي على رقم 0501234567 أو +966501234567", 0.7, False],
        ["هويتي الوطنية رقم 1234567890 وإيميلي ahmed@test.com", 0.7, True],
        ["Contact John Doe at +971 50 123 4567 or john@company.com", 0.7, False],
        ["تم العثور على تقييم طالب على جهاز يحمل رقم IMEI: 06-184755-866851-3", 0.7, False],
        ["رخصتك 78B5R2MVFAHJ48500 لا تزال مسجلة", 0.7, False]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    # Launch with appropriate settings for HF Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)
