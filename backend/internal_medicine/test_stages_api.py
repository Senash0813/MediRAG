"""
Simple test script for the stage-wise evaluation API endpoint
"""
import json
import requests


def test_evaluate_stages(query: str, base_url: str = "http://localhost:8000"):
    """Test the /evaluate-stages endpoint"""
    
    print(f"\n{'='*80}")
    print(f"Testing query: {query}")
    print('='*80)
    
    try:
        response = requests.post(
            f"{base_url}/evaluate-stages",
            json={
                "query": query,
                "verify": True,
                "temperature": 0.0
            },
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return
        
        data = response.json()
        
        # Stage A: Domain Gate
        print("\n🔍 STAGE A: Domain/Scope Gate")
        stage_a = data.get("stage_a_domain_gate", {})
        print(f"   Decision: {stage_a.get('decision')}")
        if 'top1_similarity' in stage_a:
            print(f"   Top-1 Similarity: {stage_a.get('top1_similarity')}")
            print(f"   Avg Top-K: {stage_a.get('avg_topk_similarity')}")
            print(f"   Cohesion: {stage_a.get('cohesion')}")
        
        # Stage B: RAG
        print("\n📚 STAGE B: RAG Answer Generation")
        stage_b = data.get("stage_b_rag", {})
        if "skipped" not in stage_b:
            print(f"   Retrieved Docs: {stage_b.get('num_retrieved_docs')}")
            print(f"   Top Doc Scores: {stage_b.get('top_doc_scores')}")
            answer = stage_b.get('generated_answer', '')
            print(f"   Answer Preview: {answer[:150]}...")
        else:
            print(f"   {stage_b.get('skipped')}")
        
        # Stage C: Risk Routing
        print("\n⚠️  STAGE C: Risk Routing")
        stage_c = data.get("stage_c_risk_routing", {})
        if "skipped" not in stage_c:
            print(f"   Total Sentences: {stage_c.get('total_sentences')}")
            print(f"   High-Risk Sentences: {stage_c.get('high_risk_sentences')}")
            print(f"   Risk Threshold: {stage_c.get('risk_threshold')}")
        else:
            print(f"   {stage_c.get('skipped')}")
        
        # Stage D: Verification
        print("\n✅ STAGE D: Verification")
        stage_d = data.get("stage_d_verification", {})
        if "skipped" not in stage_d:
            print(f"   Path: {stage_d.get('path')}")
            if 'nli_label' in stage_d:
                # Answer-level
                print(f"   NLI Label: {stage_d.get('nli_label')}")
                print(f"   Entail Prob: {stage_d.get('entail_prob')}")
                print(f"   Status: {stage_d.get('status')}")
            else:
                # Sentence-level
                print(f"   Verified: {stage_d.get('verified_sentences')}")
                print(f"   Regenerated: {stage_d.get('regenerated_sentences')}")
                print(f"   Hallucinated: {stage_d.get('hallucinated_sentences')}")
                print(f"   Uncertain: {stage_d.get('uncertain_sentences')}")
        else:
            print(f"   {stage_d.get('skipped')}")
        
        # Stage E: Reconstruction
        print("\n🔧 STAGE E: Reconstruction")
        stage_e = data.get("stage_e_reconstruction", {})
        if "skipped" not in stage_e:
            print(f"   Replacements Made: {stage_e.get('replacements_made')}")
        else:
            print(f"   {stage_e.get('skipped')}")
        
        # Stage F: Transparency
        print("\n📋 STAGE F: Transparency Layer")
        stage_f = data.get("stage_f_transparency", {})
        print(f"   Disclaimer Added: {stage_f.get('disclaimer_added')}")
        final = stage_f.get('final_answer', '')
        print(f"\n   Final Answer:\n   {final[:300]}...")
        
        print(f"\n{'='*80}\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Test with different types of queries
    
    test_queries = [
        "What is the treatment for type 2 diabetes?",
        "How should I take aspirin?",
        "What is quantum computing?",  # Likely out of domain
    ]
    
    print("\n" + "="*80)
    print("STAGE-WISE EVALUATION API TEST")
    print("="*80)
    
    for query in test_queries:
        test_evaluate_stages(query)
    
    print("\n✅ Test script complete!")
