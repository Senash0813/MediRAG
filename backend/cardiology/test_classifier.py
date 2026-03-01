"""
Test Script for Query Classification System
Run this to verify the classification logic works correctly
"""

import sys
sys.path.insert(0, '/c/Users/User/Desktop/MediRAG/backend/cardiology')

from query_classifier import classify_query


def test_classification():
    """Test the query classifier with various inputs"""
    
    test_cases = [
        # (query, expected_intent_keyword)
        ("Hello, how are you?", "GREETING"),
        ("Hi there!", "GREETING"),
        ("What is atrial fibrillation?", "CARDIOLOGY"),
        ("Tell me about myocardial infarction", "CARDIOLOGY"),
        ("How does a stent work?", "CARDIOLOGY"),
        ("What is the capital of France?", "GENERAL"),
        ("What should I know about python?", "GENERAL"),
        ("How do I treat schizophrenia?", "OTHER_MEDICAL"),
        ("What's the best neurology treatment?", "OTHER_MEDICAL"),
        ("Tell me about cancer therapies", "OTHER_MEDICAL"),
    ]
    
    print("\n" + "=" * 80)
    print("QUERY CLASSIFICATION TEST")
    print("=" * 80)
    
    for query, expected_keyword in test_cases:
        intent, confidence, metadata = classify_query(query, vectorstore=None)
        
        status = "✅" if expected_keyword in intent else "⚠️"
        
        print(f"\n{status} Query: \"{query}\"")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Has Cardiology Keywords: {metadata['has_cardiology_keywords']}")
        print(f"   Has Other Medical Keywords: {metadata['has_other_medical_keywords']}")


if __name__ == "__main__":
    test_classification()
    print("\n" + "=" * 80)
    print("✅ Classification test completed!")
    print("=" * 80 + "\n")
