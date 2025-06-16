
#!/usr/bin/env python3
"""
Analysis of MutazYoune/Arabic-NER-PII model from Hugging Face
Comparing approaches and extracting useful insights for our project
"""

import requests
import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class HuggingFaceModelAnalyzer:
    """Analyzer for Hugging Face Arabic PII NER models"""
    
    def __init__(self, model_name: str = "MutazYoune/Arabic-NER-PII"):
        self.model_name = model_name
        self.api_base = "https://huggingface.co/api"
        self.model_info = None
        self.tokenizer = None
        self.model = None
    
    def fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Hugging Face API"""
        try:
            # Get model info
            response = requests.get(f"{self.api_base}/models/{self.model_name}")
            if response.status_code == 200:
                self.model_info = response.json()
                return self.model_info
            else:
                print(f"Failed to fetch model info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return {}
    
    def analyze_model_config(self) -> Dict[str, Any]:
        """Analyze model configuration and architecture"""
        try:
            # Try to load tokenizer and model config
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get config without loading the full model
            config_url = f"https://huggingface.co/{self.model_name}/raw/main/config.json"
            config_response = requests.get(config_url)
            
            if config_response.status_code == 200:
                config = config_response.json()
                return config
            else:
                print(f"Could not fetch config.json: {config_response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error analyzing model config: {e}")
            return {}
    
    def get_label_mapping(self) -> Dict[str, Any]:
        """Extract label mappings from the model"""
        try:
            config = self.analyze_model_config()
            
            label_info = {}
            if 'label2id' in config:
                label_info['label2id'] = config['label2id']
            if 'id2label' in config:
                label_info['id2label'] = config['id2label']
            
            return label_info
        except Exception as e:
            print(f"Error getting label mapping: {e}")
            return {}
    
    def compare_with_our_approach(self) -> Dict[str, Any]:
        """Compare MutazYoune model with our current approach"""
        
        our_labels = [
            'O', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION',
            'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PHONE', 'I-PHONE',
            'B-EMAIL', 'I-EMAIL', 'B-ID_NUMBER', 'I-ID_NUMBER',
            'B-ADDRESS', 'I-ADDRESS'
        ]
        
        model_config = self.analyze_model_config()
        label_mapping = self.get_label_mapping()
        
        comparison = {
            "our_approach": {
                "base_model": "aubmindlab/bert-base-arabertv2",
                "num_labels": len(our_labels),
                "labels": our_labels,
                "approach": "Rule-based + ML hybrid",
                "datasets": ["Wojood", "Synthetic augmentation"],
                "features": [
                    "Multi-dialect Arabic support",
                    "Regional PII patterns (Gulf, Levant, Maghreb)",
                    "Real-time rule-based detection",
                    "Confidence scoring",
                    "Context validation"
                ]
            },
            "mutaz_approach": {
                "base_model": model_config.get('_name_or_path', 'Unknown'),
                "num_labels": model_config.get('num_labels', 'Unknown'),
                "labels": list(label_mapping.get('id2label', {}).values()) if label_mapping.get('id2label') else [],
                "architecture": model_config.get('architectures', ['Unknown'])[0] if model_config.get('architectures') else 'Unknown',
                "vocab_size": model_config.get('vocab_size', 'Unknown'),
                "hidden_size": model_config.get('hidden_size', 'Unknown'),
                "model_info": self.model_info or {}
            }
        }
        
        return comparison
    
    def test_model_inference(self, test_texts: List[str]) -> List[Dict[str, Any]]:
        """Test the HuggingFace model on sample texts"""
        try:
            # Load model for inference
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            results = []
            
            for text in test_texts:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_ids = torch.argmax(predictions, dim=-1)
                
                # Convert to labels
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                predicted_labels = [self.model.config.id2label[id.item()] for id in predicted_ids[0]]
                
                # Combine tokens and labels
                token_predictions = []
                for token, label, confidence in zip(tokens, predicted_labels, predictions[0]):
                    if token not in ['[CLS]', '[SEP]', '[PAD]']:
                        token_predictions.append({
                            "token": token,
                            "label": label,
                            "confidence": confidence[torch.argmax(confidence)].item()
                        })
                
                results.append({
                    "text": text,
                    "predictions": token_predictions
                })
            
            return results
            
        except Exception as e:
            print(f"Error during model inference: {e}")
            return []
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report"""
        
        print("🔍 ANALYZING MUTAZYOUNE/ARABIC-NER-PII MODEL")
        print("=" * 60)
        
        # Fetch model information
        self.fetch_model_info()
        
        # Get comparison data
        comparison = self.compare_with_our_approach()
        
        # Test with sample Arabic PII text
        test_texts = [
            "اسمي أحمد محمد ورقم هاتفي 0501234567",
            "يمكنك التواصل معي على البريد الإلكتروني ahmed@example.com",
            "رقم الهوية الوطنية: 1234567890",
            "العنوان: الرياض، المملكة العربية السعودية"
        ]
        
        inference_results = self.test_model_inference(test_texts)
        
        # Generate report
        report = []
        report.append("🤖 HUGGING FACE MODEL ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Model Information
        report.append("\n📊 MODEL INFORMATION:")
        report.append("-" * 40)
        mutaz_info = comparison["mutaz_approach"]
        report.append(f"Model Name: {self.model_name}")
        report.append(f"Base Model: {mutaz_info['base_model']}")
        report.append(f"Architecture: {mutaz_info['architecture']}")
        report.append(f"Number of Labels: {mutaz_info['num_labels']}")
        report.append(f"Vocabulary Size: {mutaz_info['vocab_size']}")
        report.append(f"Hidden Size: {mutaz_info['hidden_size']}")
        
        # Label Comparison
        report.append("\n🏷️  LABEL COMPARISON:")
        report.append("-" * 40)
        report.append("MutazYoune Labels:")
        for i, label in enumerate(mutaz_info['labels'][:10]):  # Show first 10
            report.append(f"  {i}: {label}")
        if len(mutaz_info['labels']) > 10:
            report.append(f"  ... and {len(mutaz_info['labels']) - 10} more")
        
        report.append("\nOur Current Labels:")
        our_info = comparison["our_approach"]
        for i, label in enumerate(our_info['labels']):
            report.append(f"  {i}: {label}")
        
        # Approach Comparison
        report.append("\n🔄 APPROACH COMPARISON:")
        report.append("-" * 40)
        report.append("MutazYoune Approach:")
        report.append("  • Pure ML-based NER model")
        report.append("  • Fine-tuned transformer architecture")
        report.append("  • Single model inference")
        
        report.append("\nOur Hybrid Approach:")
        for feature in our_info['features']:
            report.append(f"  • {feature}")
        
        # Inference Results
        if inference_results:
            report.append("\n🧪 INFERENCE TEST RESULTS:")
            report.append("-" * 40)
            for result in inference_results:
                report.append(f"\nText: {result['text']}")
                report.append("Predictions:")
                for pred in result['predictions'][:5]:  # Show first 5 tokens
                    report.append(f"  {pred['token']} -> {pred['label']} ({pred['confidence']:.3f})")
        
        # Recommendations
        report.append("\n💡 INTEGRATION RECOMMENDATIONS:")
        report.append("-" * 40)
        report.append("1. Ensemble Approach:")
        report.append("   • Use MutazYoune model as additional ML component")
        report.append("   • Combine with our rule-based detection")
        report.append("   • Weight predictions based on confidence scores")
        
        report.append("\n2. Label Mapping:")
        report.append("   • Map MutazYoune labels to our PII taxonomy")
        report.append("   • Handle label format differences (BIO vs others)")
        
        report.append("\n3. Performance Optimization:")
        report.append("   • Cache model predictions for repeated texts")
        report.append("   • Use batch inference for better throughput")
        report.append("   • Implement model selection based on text characteristics")
        
        return "\n".join(report)

def main():
    """Main analysis function"""
    analyzer = HuggingFaceModelAnalyzer("MutazYoune/Arabic-NER-PII")
    
    try:
        report = analyzer.generate_comparison_report()
        print(report)
        
        # Save report to file
        with open("mutazyoune_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: mutazyoune_analysis_report.txt")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
