from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
from rules import PIIDetector, PIIMatch
from performance_optimizer import performance_monitor

app = FastAPI(title="PII Detection Engine", description="Arabic and English PII detection system")

detector = PIIDetector()

class TextInput(BaseModel):
    text: str
    min_confidence: float = 0.7

class PIIResponse(BaseModel):
    original_text: str
    masked_text: str
    detected_pii: List[Dict[str, Any]]
    summary: Dict[str, int]

def mask_pii_in_text(text: str, pii_matches: List[PIIMatch]) -> str:
    """Replace detected PII with [MASKED] tokens as per competition format"""
    if not pii_matches:
        return text

    # Sort matches by start position in reverse order to avoid index shifting
    sorted_matches = sorted(pii_matches, key=lambda x: x.start_pos, reverse=True)

    masked_text = text
    for match in sorted_matches:
        # Use competition standard [MASKED] format
        mask = "[MASKED]"
        masked_text = masked_text[:match.start_pos] + mask + masked_text[match.end_pos:]

    return masked_text

@app.get("/", response_class=HTMLResponse)
async def get_test_interface():
    """Serve the interactive test interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PII Detection Engine Tester</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                font-family: monospace;
                resize: vertical;
            }
            input[type="range"] {
                width: 100%;
                margin: 10px 0;
            }
            .confidence-display {
                text-align: center;
                font-weight: bold;
                color: #666;
            }
            button {
                background: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            button:hover {
                background: #0056b3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }
            .masked-text {
                background: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
                white-space: pre-wrap;
                margin: 10px 0;
            }
            .detected-pii {
                margin-top: 15px;
            }
            .pii-item {
                background: white;
                padding: 10px;
                margin: 5px 0;
                border-radius: 3px;
                border-left: 3px solid #28a745;
            }
            .summary {
                background: #d4edda;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #fff3cd;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }
            .example-text {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 3px;
                font-family: monospace;
                margin: 10px 0;
                cursor: pointer;
                border: 1px solid #ddd;
            }
            .example-text:hover {
                background: #e9ecef;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 PII Detection Engine Tester</h1>

            <div class="form-group">
                <label for="textInput">Enter text to test for PII detection:</label>
                <textarea id="textInput" placeholder="Enter Arabic or English text containing potential PII like phone numbers, emails, IBANs, etc."></textarea>
            </div>

            <div class="form-group">
                <label for="confidenceSlider">Minimum Confidence Threshold:</label>
                <input type="range" id="confidenceSlider" min="0.1" max="1.0" step="0.1" value="0.7">
                <div class="confidence-display">0.7</div>
            </div>

            <button onclick="testPIIDetection()" id="testButton">🔍 Detect PII</button>

            <div class="loading" id="loading">
                <p>🔄 Processing...</p>
            </div>

            <div id="results"></div>

            <div class="examples">
                <h3>📋 Test Examples (Click to use):</h3>
                <div class="example-text" onclick="useExample(this)">
اتصل بي على رقم 0501234567 أو +966501234567
إيميلي هو ahmed@example.com
رقم الهوية الوطنية: 1234567890
الآيبان: SA12 3456 7890 1234 5678 9012 3456
                </div>
                <div class="example-text" onclick="useExample(this)">
Contact me at john.doe@company.org
Phone: +962 77 123 4567
Credit Card: 4111-1111-1111-1111
IBAN: JO94 CBJO 0010 0000 0000 0131 0003
                </div>
                <div class="example-text" onclick="useExample(this)">
UAE Mobile: +971 50 123 4567
Emirates ID: 784-2000-1234567-8
Egypt Phone: +20 10 1234 5678
Mixed: call 05 1 234 5678 or email test@domain.com
                </div>
            </div>
        </div>

        <script>
            const confidenceSlider = document.getElementById('confidenceSlider');
            const confidenceDisplay = document.querySelector('.confidence-display');

            confidenceSlider.addEventListener('input', function() {
                confidenceDisplay.textContent = this.value;
            });

            function useExample(element) {
                document.getElementById('textInput').value = element.textContent.trim();
            }

            async function testPIIDetection() {
                const textInput = document.getElementById('textInput').value;
                const confidence = parseFloat(document.getElementById('confidenceSlider').value);
                const resultsDiv = document.getElementById('results');
                const testButton = document.getElementById('testButton');
                const loading = document.getElementById('loading');

                if (!textInput.trim()) {
                    resultsDiv.innerHTML = '<div class="error">Please enter some text to test.</div>';
                    return;
                }

                testButton.disabled = true;
                loading.style.display = 'block';
                resultsDiv.innerHTML = '';

                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: textInput,
                            min_confidence: confidence
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                } finally {
                    testButton.disabled = false;
                    loading.style.display = 'none';
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');

                let html = '<div class="results">';
                html += '<h3>🎯 Detection Results</h3>';

                // Summary
                html += '<div class="summary">';
                html += '<h4>📊 Summary</h4>';
                html += '<ul>';
                for (const [piiType, count] of Object.entries(data.summary)) {
                    html += `<li><strong>${piiType}:</strong> ${count} detected</li>`;
                }
                html += '</ul>';
                html += '</div>';

                // Masked text
                html += '<h4>🔒 Masked Text</h4>';
                html += `<div class="masked-text">${escapeHtml(data.masked_text)}</div>`;

                // Detected PII details
                if (data.detected_pii.length > 0) {
                    html += '<div class="detected-pii">';
                    html += '<h4>🏷️ Detected PII Details</h4>';

                    data.detected_pii.forEach((pii, index) => {
                        html += `<div class="pii-item">`;
                        html += `<strong>Type:</strong> ${pii.pii_type}<br>`;
                        html += `<strong>Text:</strong> "${escapeHtml(pii.text)}"<br>`;
                        html += `<strong>Pattern:</strong> ${pii.pattern_name}<br>`;
                        html += `<strong>Confidence:</strong> ${(pii.confidence * 100).toFixed(1)}%<br>`;
                        html += `<strong>Position:</strong> ${pii.start_pos}-${pii.end_pos}`;
                        html += '</div>';
                    });

                    html += '</div>';
                } else {
                    html += '<p><em>No PII detected with the current confidence threshold.</em></p>';
                }

                html += '</div>';
                resultsDiv.innerHTML = html;
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // Allow testing with Enter key
            document.getElementById('textInput').addEventListener('keydown', function(event) {
                if (event.ctrlKey && event.key === 'Enter') {
                    testPIIDetection();
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/detect", response_model=PIIResponse)
async def detect_pii(input_data: TextInput):
    """Detect PII in the provided text using rule-based approach"""
    try:
        # Detect all PII in the text
        detected_pii = detector.detect_all_pii(input_data.text, input_data.min_confidence)

        # Create masked version
        masked_text = mask_pii_in_text(input_data.text, detected_pii)

        # Convert PIIMatch objects to dictionaries
        pii_list = []
        for match in detected_pii:
            pii_list.append({
                "text": match.text,
                "pii_type": match.pii_type,
                "start_pos": match.start_pos,
                "end_pos": match.end_pos,
                "confidence": match.confidence,
                "pattern_name": match.pattern_name,
                "detection_method": "rules"
            })

        # Create summary
        summary = {}
        for match in detected_pii:
            summary[match.pii_type] = summary.get(match.pii_type, 0) + 1

        return PIIResponse(
            original_text=input_data.text,
            masked_text=masked_text,
            detected_pii=pii_list,
            summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting PII: {str(e)}")

@app.post("/detect-ensemble", response_model=PIIResponse)
async def detect_pii_ensemble(input_data: TextInput):
    """Detect PII using ensemble approach (rules + MutazYoune model)"""
    if not ensemble_available:
        raise HTTPException(
            status_code=503, 
            detail="Ensemble detection not available. MutazYoune model could not be loaded."
        )

    try:
        # Get ensemble predictions
        ensemble_predictions = ensemble_detector.detect_ensemble_pii(
            input_data.text, 
            input_data.min_confidence
        )

        # Convert to standard format for masking
        detected_pii = []
        for pred in ensemble_predictions:
            # Create a PIIMatch-like object for compatibility with masking function
            class EnsembleMatch:
                def __init__(self, pred):
                    self.text = pred.text
                    self.pii_type = pred.pii_type
                    self.start_pos = pred.start_pos
                    self.end_pos = pred.end_pos
                    self.confidence = pred.confidence
                    self.pattern_name = f"ensemble-{'-'.join(pred.source_models)}"

            detected_pii.append(EnsembleMatch(pred))

        # Create masked version
        masked_text = mask_pii_in_text(input_data.text, detected_pii)

        # Convert to API response format
        pii_list = []
        for pred in ensemble_predictions:
            pii_list.append({
                "text": pred.text,
                "pii_type": pred.pii_type,
                "start_pos": pred.start_pos,
                "end_pos": pred.end_pos,
                "confidence": pred.confidence,
                "detection_method": "ensemble",
                "source_models": pred.source_models,
                "individual_scores": pred.individual_scores
            })

        # Create summary
        summary = {}
        for pred in ensemble_predictions:
            summary[pred.pii_type] = summary.get(pred.pii_type, 0) + 1

        return PIIResponse(
            original_text=input_data.text,
            masked_text=masked_text,
            detected_pii=pii_list,
            summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ensemble detection: {str(e)}")

@app.post("/detect-batch")
async def detect_pii_batch(texts: List[str], min_confidence: float = 0.7):
    """Batch PII detection for multiple texts"""
    try:
        if len(texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        results = []
        for text in texts:
            detected_pii = detector.detect_all_pii(text, min_confidence)
            masked_text = mask_pii_in_text(text, detected_pii)
            
            pii_list = []
            for match in detected_pii:
                pii_list.append({
                    "text": match.text,
                    "pii_type": match.pii_type,
                    "start_pos": match.start_pos,
                    "end_pos": match.end_pos,
                    "confidence": match.confidence,
                    "pattern_name": match.pattern_name,
                    "detection_method": "rules"
                })
            
            summary = {}
            for match in detected_pii:
                summary[match.pii_type] = summary.get(match.pii_type, 0) + 1
            
            results.append({
                "original_text": text,
                "masked_text": masked_text,
                "detected_pii": pii_list,
                "summary": summary
            })
        
        return {"results": results, "processed_count": len(texts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch detection: {str(e)}")

@app.get("/performance-stats")
async def get_performance_stats():
    """Get performance statistics"""
    try:
        stats = performance_monitor.get_stats()
        return {"performance_metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/mask")
async def mask_text(input_data: TextInput):
    """Legacy endpoint for simple text masking"""
    try:
        # Detect PII
        detected_pii = detector.detect_all_pii(input_data.text, input_data.min_confidence)

        # Return masked text
        masked_text = mask_pii_in_text(input_data.text, detected_pii)

        return {"masked_text": masked_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error masking text: {str(e)}")

@app.get("/test-enhanced-detection")
async def test_enhanced_detection_endpoint():
    """Comprehensive test endpoint for all enhanced PII detection features"""
    test_cases = [
        # Credit cards
        "بطاقتي الائتمانية 4111-1111-1111-1111 visa card",
        "Credit card number: 5555555555554444 MasterCard",
        "AmEx: 378282246310005",
        "Invalid card: 1234567890123456",

        # Mixed Arabic-English context
        "اتصل بي على رقم +966501234567 أو email me at ahmed@test.com",
        "هويتي الوطنية رقم 1234567890 وبطاقة visa 4111111111111111",

        # Regional dialects and variations
        "أرقام مختلفة: ٠٥٠١٢٣٤٥٦٧ و ۰۵۰۱۲۳۴۵۶۷",
        "Different numbers: 05-012-34567 and +966 50 123 4567",

        # Context with negation
        "This is not a real credit card: 4111111111111111",
        "هذا مثال وهمي: +966501234567",

        # High-confidence context
        "لطفاً اتصل بي على هاتفي 0501234567",
        "Please contact me at my phone number +966501234567",
    ]

    results = []
    for test_text in test_cases:
        matches = detector.detect_all_pii(test_text, min_confidence=0.5)
        results.append({
            "test_text": test_text,
            "detected_count": len(matches),
            "matches": [{
                "text": m.text,
                "type": m.pii_type,
                "confidence": round(m.confidence, 3),
                "pattern": m.pattern_name,
                "position": f"{m.start_pos}-{m.end_pos}"
            } for m in matches]
        })

    return {"enhanced_test_results": results}

@app.get("/test-saudi-mobile")
async def test_saudi_mobile_endpoint():
    """Test endpoint specifically for Saudi mobile numbers"""
    test_cases = [
        "+966 50 123 4567",
        "+966501234567",
        "00966 50 123 4567",
        "00966501234567", 
        "05 0 123 4567",
        "050 123 4567",
        "0501234567",
        "+966 55 999 8888",
        "Invalid: +966 40 123 4567",  # Invalid prefix
        "Invalid: 0601234567",        # Invalid prefix
    ]

    results = []
    for test_text in test_cases:
        matches = detector.detect_saudi_mobile_numbers(test_text)
        results.append({
            "test_text": test_text,
            "detected": len(matches) > 0,
            "matches": [{"text": m.text, "confidence": m.confidence, "pattern": m.pattern_name} for m in matches]
        })

    return {"test_results": results}

@app.get("/test-competition")
async def test_competition_examples():
    """Test with official competition examples"""
    test_cases = [
        # Official examples from challenge
        "يعمل احمد محمد في شركة تقنية",
        "للتواصل مع سارة على الرقم 0501234567",
        "تم العثور على تقييم طالب على جهاز يحمل رقم IMEI: 06-184755-866851-3",
        "أنا جاريك موراي. سأقوم باستلام نتائج الفحص",
        "عزيزتي كارمن، بصفتك مهندسة العلامة التجارية الوطنية لدينا",
        "الطفل البالغ من العمر 88 عامًا في 5862",
        "IPv6 edaf:fd8f:e1e8:cfec:8bab:1afd:6aad:550c",
        "رخصتك 78B5R2MVFAHJ48500 لا تزال مسجلة",
    ]
    
    results = []
    for test_text in test_cases:
        detected_pii = detector.detect_all_pii(test_text, min_confidence=0.7)
        masked_text = mask_pii_in_text(test_text, detected_pii)
        
        results.append({
            "source": test_text,
            "target": masked_text,
            "detected_pii_count": len(detected_pii),
            "detected_types": [match.pii_type for match in detected_pii]
        })
    
    return {"competition_test_results": results}

if __name__ == "__main__":
    import uvicorn
    # Assuming ensemble_available and ensemble_detector are defined elsewhere (e.g., in rules.py)
    ensemble_available = False # Replace with actual check if the ensemble model is loaded
    class EnsembleDetector:
        def detect_ensemble_pii(self, text, min_confidence):
            return []
    ensemble_detector = EnsembleDetector()

    uvicorn.run(app, host="0.0.0.0", port=5000)