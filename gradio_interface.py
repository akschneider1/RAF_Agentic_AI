
import gradio as gr
import requests
import json

def detect_arabic_pii(text, confidence_threshold=0.7, use_obfuscation=False):
    """Gradio interface for Arabic PII detection"""
    try:
        # Call your existing FastAPI endpoint
        response = requests.post(
            "http://localhost:5000/detect",
            json={
                "text": text,
                "min_confidence": confidence_threshold,
                "use_obfuscation": use_obfuscation
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Format results for Gradio
            masked_text = result["masked_text"]
            summary = result["summary"]
            
            # Create summary text
            summary_text = "**Detected PII:**\n"
            for pii_type, count in summary.items():
                summary_text += f"- {pii_type}: {count}\n"
            
            # Create detailed results
            details = "**Detection Details:**\n"
            for pii in result["detected_pii"]:
                details += f"- **{pii['pii_type']}**: '{pii['text']}' (Confidence: {pii['confidence']:.2f})\n"
            
            return masked_text, summary_text, details
        else:
            return "Error processing text", "Failed to detect PII", "Please try again"
            
    except Exception as e:
        return f"Error: {str(e)}", "System error", "Please check the system"

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
        ["Contact John Doe at +971 50 123 4567 or john@company.com", 0.7, False]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # Start Gradio on port 7860 (standard for HF Spaces)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
