
#!/usr/bin/env python3
"""
Comprehensive comparison study between our hybrid approach 
and MutazYoune/Arabic-NER-PII model
"""

import time
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from rules import PIIDetector
from model_ensemble import MutazYouneIntegration, AdvancedEnsembleDetector

class ComprehensiveModelComparison:
    """Compare different PII detection approaches"""
    
    def __init__(self):
        self.rule_detector = PIIDetector()
        self.mutaz_detector = MutazYouneIntegration()
        self.ensemble_detector = AdvancedEnsembleDetector()
        
        # Test datasets
        self.test_cases = self._create_test_cases()
        
        # Results storage
        self.results = {
            'rule_based': [],
            'mutazyoune': [],
            'ensemble': [],
            'performance_metrics': {},
            'analysis_summary': {}
        }
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases with ground truth"""
        
        test_cases = [
            # Simple Arabic names
            {
                "text": "اسمي أحمد محمد الفهد",
                "expected_pii": [
                    {"type": "PERSON", "text": "أحمد محمد الفهد", "high_confidence": True}
                ],
                "category": "simple_names"
            },
            
            # Phone numbers (various formats)
            {
                "text": "اتصل بي على رقم 0501234567 أو +966501234567",
                "expected_pii": [
                    {"type": "PHONE", "text": "0501234567", "high_confidence": True},
                    {"type": "PHONE", "text": "+966501234567", "high_confidence": True}
                ],
                "category": "phone_numbers"
            },
            
            # Email addresses
            {
                "text": "راسلني على البريد الإلكتروني ahmed.mohammed@ksu.edu.sa",
                "expected_pii": [
                    {"type": "EMAIL", "text": "ahmed.mohammed@ksu.edu.sa", "high_confidence": True}
                ],
                "category": "email_addresses"
            },
            
            # Complex mixed content
            {
                "text": "الدكتور عبدالله بن سعد يعمل في جامعة الملك سعود ويمكن التواصل معه على 0112345678",
                "expected_pii": [
                    {"type": "PERSON", "text": "عبدالله بن سعد", "high_confidence": True},
                    {"type": "ORGANIZATION", "text": "جامعة الملك سعود", "high_confidence": True},
                    {"type": "PHONE", "text": "0112345678", "high_confidence": True}
                ],
                "category": "complex_mixed"
            },
            
            # National IDs
            {
                "text": "رقم الهوية الوطنية الخاص بي هو 1234567890",
                "expected_pii": [
                    {"type": "ID_NUMBER", "text": "1234567890", "high_confidence": True}
                ],
                "category": "national_ids"
            },
            
            # Addresses
            {
                "text": "أسكن في الرياض، حي النخيل، شارع الملك فهد، رقم المبنى 123",
                "expected_pii": [
                    {"type": "LOCATION", "text": "الرياض", "high_confidence": True},
                    {"type": "LOCATION", "text": "حي النخيل", "high_confidence": False},
                    {"type": "ADDRESS", "text": "شارع الملك فهد", "high_confidence": True}
                ],
                "category": "addresses"
            },
            
            # Organization names
            {
                "text": "أعمل في شركة أرامكو السعودية ومقرها في الظهران",
                "expected_pii": [
                    {"type": "ORGANIZATION", "text": "أرامكو السعودية", "high_confidence": True},
                    {"type": "LOCATION", "text": "الظهران", "high_confidence": True}
                ],
                "category": "organizations"
            },
            
            # Mixed Arabic-English
            {
                "text": "Contact Dr. Ahmad Al-Rashid at +966501234567 or ahmad.rashid@kfupm.edu.sa",
                "expected_pii": [
                    {"type": "PERSON", "text": "Ahmad Al-Rashid", "high_confidence": True},
                    {"type": "PHONE", "text": "+966501234567", "high_confidence": True},
                    {"type": "EMAIL", "text": "ahmad.rashid@kfupm.edu.sa", "high_confidence": True}
                ],
                "category": "mixed_language"
            },
            
            # False positive challenges
            {
                "text": "الرقم 0501234567 ليس رقم هاتف حقيقي، هذا مجرد مثال",
                "expected_pii": [
                    {"type": "PHONE", "text": "0501234567", "high_confidence": False}  # Should have lower confidence due to negation
                ],
                "category": "false_positive_test"
            },
            
            # Credit cards (synthetic for testing)
            {
                "text": "رقم البطاقة الائتمانية: 4111-1111-1111-1111 (للاختبار فقط)",
                "expected_pii": [
                    {"type": "CREDIT_CARD", "text": "4111-1111-1111-1111", "high_confidence": True}
                ],
                "category": "credit_cards"
            }
        ]
        
        return test_cases
    
    def run_model_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of all models"""
        
        print("🔬 COMPREHENSIVE MODEL COMPARISON STUDY")
        print("=" * 60)
        
        total_tests = len(self.test_cases)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📝 Test {i}/{total_tests} ({test_case['category']})")
            print(f"Text: {test_case['text'][:50]}...")
            
            # Test rule-based detector
            rule_start = time.time()
            rule_matches = self.rule_detector.detect_all_pii(test_case['text'], 0.5)
            rule_time = time.time() - rule_start
            
            # Test MutazYoune detector
            mutaz_start = time.time()
            mutaz_predictions = self.mutaz_detector.predict(test_case['text'], 0.5)
            mutaz_time = time.time() - mutaz_start
            
            # Test ensemble detector
            ensemble_start = time.time()
            ensemble_predictions = self.ensemble_detector.detect_ensemble_pii(test_case['text'], 0.5)
            ensemble_time = time.time() - ensemble_start
            
            # Store results
            self.results['rule_based'].append({
                'test_case': test_case,
                'predictions': [self._convert_rule_match(m) for m in rule_matches],
                'execution_time': rule_time,
                'prediction_count': len(rule_matches)
            })
            
            self.results['mutazyoune'].append({
                'test_case': test_case,
                'predictions': [self._convert_mutaz_prediction(p) for p in mutaz_predictions],
                'execution_time': mutaz_time,
                'prediction_count': len(mutaz_predictions)
            })
            
            self.results['ensemble'].append({
                'test_case': test_case,
                'predictions': [self._convert_ensemble_prediction(p) for p in ensemble_predictions],
                'execution_time': ensemble_time,
                'prediction_count': len(ensemble_predictions)
            })
            
            print(f"   Rules: {len(rule_matches)} detections ({rule_time:.3f}s)")
            print(f"   MutazYoune: {len(mutaz_predictions)} detections ({mutaz_time:.3f}s)")
            print(f"   Ensemble: {len(ensemble_predictions)} detections ({ensemble_time:.3f}s)")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Generate analysis summary
        self._generate_analysis_summary()
        
        return self.results
    
    def _convert_rule_match(self, match) -> Dict[str, Any]:
        """Convert rule match to standard format"""
        return {
            'text': match.text,
            'type': match.pii_type,
            'confidence': match.confidence,
            'start': match.start_pos,
            'end': match.end_pos,
            'source': 'rules'
        }
    
    def _convert_mutaz_prediction(self, pred) -> Dict[str, Any]:
        """Convert MutazYoune prediction to standard format"""
        return {
            'text': pred.text,
            'type': pred.pii_type,
            'confidence': pred.confidence,
            'start': pred.start_pos,
            'end': pred.end_pos,
            'source': 'mutazyoune'
        }
    
    def _convert_ensemble_prediction(self, pred) -> Dict[str, Any]:
        """Convert ensemble prediction to standard format"""
        return {
            'text': pred.text,
            'type': pred.pii_type,
            'confidence': pred.confidence,
            'start': pred.start_pos,
            'end': pred.end_pos,
            'source': 'ensemble',
            'source_models': pred.source_models
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for each approach"""
        
        metrics = {}
        
        for approach in ['rule_based', 'mutazyoune', 'ensemble']:
            results = self.results[approach]
            
            total_predictions = sum(r['prediction_count'] for r in results)
            total_time = sum(r['execution_time'] for r in results)
            avg_time_per_test = total_time / len(results)
            
            # Calculate precision/recall approximation
            tp = fp = fn = 0
            
            for result in results:
                expected = result['test_case']['expected_pii']
                predicted = result['predictions']
                
                # Simple overlap-based matching
                for exp in expected:
                    found = any(
                        pred['type'] == exp['type'] and 
                        exp['text'].lower() in pred['text'].lower()
                        for pred in predicted
                    )
                    if found:
                        tp += 1
                    else:
                        fn += 1
                
                # Count false positives (predictions not in expected)
                for pred in predicted:
                    found = any(
                        pred['type'] == exp['type'] and 
                        exp['text'].lower() in pred['text'].lower()
                        for exp in expected
                    )
                    if not found:
                        fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[approach] = {
                'total_predictions': total_predictions,
                'total_time': total_time,
                'avg_time_per_test': avg_time_per_test,
                'predictions_per_second': total_predictions / total_time if total_time > 0 else 0,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        self.results['performance_metrics'] = metrics
    
    def _generate_analysis_summary(self):
        """Generate comprehensive analysis summary"""
        
        metrics = self.results['performance_metrics']
        
        # Performance comparison
        best_f1 = max(metrics.keys(), key=lambda k: metrics[k]['f1_score'])
        fastest = min(metrics.keys(), key=lambda k: metrics[k]['avg_time_per_test'])
        most_precise = max(metrics.keys(), key=lambda k: metrics[k]['precision'])
        highest_recall = max(metrics.keys(), key=lambda k: metrics[k]['recall'])
        
        # Category analysis
        category_performance = defaultdict(lambda: defaultdict(int))
        
        for approach in ['rule_based', 'mutazyoune', 'ensemble']:
            for result in self.results[approach]:
                category = result['test_case']['category']
                category_performance[approach][category] = result['prediction_count']
        
        summary = {
            'best_overall_f1': best_f1,
            'fastest_approach': fastest,
            'most_precise': most_precise,
            'highest_recall': highest_recall,
            'category_performance': dict(category_performance),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        self.results['analysis_summary'] = summary
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Performance-based recommendations
        ensemble_f1 = metrics['ensemble']['f1_score']
        rule_f1 = metrics['rule_based']['f1_score']
        mutaz_f1 = metrics['mutazyoune']['f1_score']
        
        if ensemble_f1 > max(rule_f1, mutaz_f1):
            recommendations.append("✅ Ensemble approach shows best overall performance - recommend for production")
        
        if metrics['rule_based']['precision'] > 0.9:
            recommendations.append("✅ Rule-based detector has high precision - excellent for high-confidence structured PII")
        
        if metrics['mutazyoune']['recall'] > metrics['rule_based']['recall']:
            recommendations.append("✅ MutazYoune model has better recall - good for catching missed entities")
        
        # Speed considerations
        if metrics['rule_based']['avg_time_per_test'] < metrics['mutazyoune']['avg_time_per_test']:
            recommendations.append("⚡ Rule-based detection is faster - consider for real-time applications")
        
        # Specific use case recommendations
        recommendations.extend([
            "📱 Use rule-based for phone numbers, emails, and structured identifiers",
            "🤖 Use MutazYoune for Arabic person/organization names and locations",
            "🔄 Use ensemble for comprehensive coverage with confidence scoring",
            "📊 Implement dynamic model selection based on text characteristics"
        ])
        
        return recommendations
    
    def create_performance_visualization(self, save_path: str = "model_comparison_charts.png"):
        """Create visualization of model performance"""
        
        metrics = self.results['performance_metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1 Score comparison
        approaches = list(metrics.keys())
        f1_scores = [metrics[app]['f1_score'] for app in approaches]
        
        ax1.bar(approaches, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('F1 Score Comparison')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        
        # Precision vs Recall
        precisions = [metrics[app]['precision'] for app in approaches]
        recalls = [metrics[app]['recall'] for app in approaches]
        
        ax2.scatter(precisions, recalls, s=100, alpha=0.7)
        for i, app in enumerate(approaches):
            ax2.annotate(app, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Precision')
        ax2.set_ylabel('Recall')
        ax2.set_title('Precision vs Recall')
        ax2.grid(True, alpha=0.3)
        
        # Execution Time comparison
        times = [metrics[app]['avg_time_per_test'] * 1000 for app in approaches]  # Convert to ms
        
        ax3.bar(approaches, times, color=['orange', 'purple', 'brown'])
        ax3.set_title('Average Execution Time per Test')
        ax3.set_ylabel('Time (milliseconds)')
        
        # Predictions per category
        category_data = self.results['analysis_summary']['category_performance']
        categories = list(set(cat for app_data in category_data.values() for cat in app_data.keys()))
        
        x = np.arange(len(categories))
        width = 0.25
        
        for i, app in enumerate(approaches):
            counts = [category_data[app].get(cat, 0) for cat in categories]
            ax4.bar(x + i * width, counts, width, label=app, alpha=0.8)
        
        ax4.set_xlabel('PII Categories')
        ax4.set_ylabel('Predictions Count')
        ax4.set_title('Predictions by Category')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(categories, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance visualization saved to: {save_path}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive comparison report"""
        
        metrics = self.results['performance_metrics']
        summary = self.results['analysis_summary']
        
        report = []
        report.append("🔍 COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("=" * 80)
        
        # Executive Summary
        report.append("\n📋 EXECUTIVE SUMMARY:")
        report.append("-" * 50)
        report.append(f"Best Overall Performance: {summary['best_overall_f1'].upper()}")
        report.append(f"Fastest Approach: {summary['fastest_approach'].upper()}")
        report.append(f"Most Precise: {summary['most_precise'].upper()}")
        report.append(f"Highest Recall: {summary['highest_recall'].upper()}")
        
        # Detailed Metrics
        report.append("\n📊 DETAILED PERFORMANCE METRICS:")
        report.append("-" * 50)
        
        header = f"{'Approach':<15} {'F1':<6} {'Precision':<9} {'Recall':<6} {'Time(ms)':<8} {'Pred/s':<7}"
        report.append(header)
        report.append("-" * len(header))
        
        for approach, metric in metrics.items():
            line = (f"{approach.upper():<15} "
                   f"{metric['f1_score']:<6.3f} "
                   f"{metric['precision']:<9.3f} "
                   f"{metric['recall']:<6.3f} "
                   f"{metric['avg_time_per_test']*1000:<8.1f} "
                   f"{metric['predictions_per_second']:<7.1f}")
            report.append(line)
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        report.append("-" * 50)
        for i, rec in enumerate(summary['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        # Integration Strategy
        report.append("\n🔧 INTEGRATION STRATEGY:")
        report.append("-" * 50)
        report.append("1. PRIMARY USE CASES:")
        report.append("   • High-precision structured PII (phone, email, ID): Rules-based")
        report.append("   • Arabic NER (names, organizations): MutazYoune model")
        report.append("   • Comprehensive coverage: Ensemble approach")
        
        report.append("\n2. DEPLOYMENT RECOMMENDATIONS:")
        report.append("   • Real-time API: Start with rules, fallback to ensemble")
        report.append("   • Batch processing: Use ensemble for maximum coverage")
        report.append("   • High-security contexts: Prioritize precision (rules + validation)")
        
        report.append("\n3. FUTURE IMPROVEMENTS:")
        report.append("   • Train domain-specific models on combined datasets")
        report.append("   • Implement confidence-based model routing")
        report.append("   • Add post-processing validation layers")
        
        return "\n".join(report)

def main():
    """Run comprehensive model comparison"""
    
    comparator = ComprehensiveModelComparison()
    
    # Run comparison
    results = comparator.run_model_comparison()
    
    # Generate visualization
    comparator.create_performance_visualization()
    
    # Generate report
    report = comparator.generate_comprehensive_report()
    print("\n" + report)
    
    # Save results
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        # Convert results to JSON-serializable format
        json_results = {
            'performance_metrics': results['performance_metrics'],
            'analysis_summary': results['analysis_summary']
        }
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    # Save report
    with open("model_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📄 Results saved to:")
    print(f"   • model_comparison_results.json")
    print(f"   • model_comparison_report.txt")
    print(f"   • model_comparison_charts.png")

if __name__ == "__main__":
    main()
