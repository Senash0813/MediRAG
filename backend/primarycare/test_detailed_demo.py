"""
Test script for the detailed pipeline demo endpoint.
Run this after starting the FastAPI server with: uvicorn main:app --reload
"""

import requests
import json
from typing import Dict, Any


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"=== {title} ===")
    print("=" * 80)


def format_dict_for_display(data: Dict[str, Any], indent: int = 0) -> str:
    """Format dictionary for readable display."""
    lines = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_dict_for_display(value, indent + 1))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{prefix}{key}: (showing first item)")
                lines.append(format_dict_for_display(value[0], indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, str) and len(value) > 200:
            lines.append(f"{prefix}{key}: {value[:200]}...")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def test_detailed_pipeline(
    query: str = "What psychiatric symptoms occur in anti-NMDA receptor encephalitis?",
    semantic_n: int = 10,
    final_k: int = 5,
    base_url: str = "http://localhost:8000"
) -> None:
    """Test the detailed pipeline endpoint and display results."""
    
    print_section_header("Query")
    print(f"Query: {query}")
    print(f"Semantic N: {semantic_n}")
    print(f"Final K: {final_k}")
    
    # Make request
    url = f"{base_url}/retrieve/detailed"
    payload = {
        "query": query,
        "semantic_n": semantic_n,
        "final_k": final_k
    }
    
    print(f"\nSending request to: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Display results by stage
    
    # 1. Scope Gate
    print_section_header("Scope Gate")
    scope = result.get("scope_gate", {})
    print(f"In Scope: {scope.get('in_scope')}")
    print(f"Best Specialty: {scope.get('best_specialty')}")
    print(f"Score: {scope.get('score'):.3f}")
    print(f"Threshold: {scope.get('threshold'):.3f}")
    
    if not scope.get('in_scope'):
        print("\n⚠️  Query is out of scope. Pipeline stopped.")
        return
    
    # 2. Retrieval
    print_section_header("Retrieval")
    retrieval = result.get("retrieval", [])
    print(f"Retrieved {len(retrieval)} documents")
    for i, doc in enumerate(retrieval[:3], 1):
        print(f"\n{i}. Title: {doc['title'][:80]}...")
        print(f"   Paper ID: {doc['paper_id']}")
        print(f"   Semantic Score: {doc['semantic_score']:.3f}")
    
    if len(retrieval) > 3:
        print(f"\n... and {len(retrieval) - 3} more documents")
    
    # 3. Phase-1: Semantic Candidate Retrieval
    print_section_header("Phase-1: Semantic Candidate Retrieval")
    phase1 = result.get("phase1_validation", [])
    print(f"Retrieved {len(phase1)} candidate documents")
    print("\nOutput: Raw candidate docs + semantic_score + title_similarity")
    print("-" * 80)
    for i, doc in enumerate(phase1[:5], 1):
        print(f"\n{i}. {doc['title'][:70]}...")
        print(f"   Paper ID: {doc['paper_id']}")
        print(f"   Semantic Score: {doc['semantic_score']:.3f}")
        print(f"   Title Similarity: {doc['title_similarity']:.3f}")
    
    if len(phase1) > 5:
        print(f"\n... and {len(phase1) - 5} more documents")
    
    # 4. Phase-2: Metadata Verification + Quality Scoring
    print_section_header("Phase-2: Metadata Verification + Quality Scoring")
    phase2 = result.get("phase2_validation", [])
    print(f"Final {len(phase2)} documents after verification and scoring")
    print("\nOutput: Phase-1 data + verified metadata + quality scores + risk flags")
    print("-" * 80)
    
    for i, doc in enumerate(phase2, 1):
        print(f"\n{i}. {doc['title'][:70]}...")
        print(f"   Paper ID: {doc['paper_id']}")
        
        print("\n   [FROM PHASE-1]")
        print(f"   - Semantic Score: {doc['semantic_score']:.3f}")
        print(f"   - Title Similarity: {doc['title_similarity']:.3f}")
        
        print("\n   [VERIFIED METADATA]")
        print(f"   - Citations: {doc['citation_count']} (Influential: {doc['influential_citation_count']})")
        print(f"   - Year: {doc.get('year', 'N/A')}")
        print(f"   - Publication Types: {', '.join(doc.get('publication_types', []))}")
        print(f"   - Title Match: {doc['title_match']}")
        print(f"   - External ID Match: {doc['external_id_match']}")
        print(f"   - Authority Level: {doc['authority_level']}")
        print(f"   - Freshness: {doc['freshness']}")
        print(f"   - Influential: {doc['influential']}")
        
        print("\n   [QUALITY SCORES]")
        print(f"   - Evidence Score: {doc['evidence_score']:.3f}")
        print(f"   - Final Score: {doc['final_score']:.3f}")
        print(f"   - Quality Tier: {doc['quality_tier']}")
        
        print("\n   [RISK FLAGS]")
        if doc['risk_flags']:
            print(f"   ⚠️  {', '.join(doc['risk_flags'])}")
        else:
            print("   ✓ No risk flags")
    
    # 5. Verification Level
    print_section_header("Verification Level")
    verification_level = result.get("verification_level", 1)
    print(f"Computed Verification Level: {verification_level} (1-4 scale)")
    level_descriptions = {
        1: "Low confidence - weak or limited evidence",
        2: "Moderate confidence - acceptable evidence quality",
        3: "Good confidence - strong evidence with minor limitations",
        4: "High confidence - excellent evidence quality"
    }
    print(f"Description: {level_descriptions.get(verification_level, 'Unknown')}")
    
    # 6. Instructor Prompt
    print_section_header("Instructor Prompt")
    instructor = result.get("instructor_prompt", {})
    print(f"Verification Level: {instructor.get('verification_level')}")
    print(f"Answer Mode: {instructor.get('answer_mode')}")
    print(f"Required Sections: {', '.join(instructor.get('required_sections', []))}")
    print("\nConstraints:")
    for constraint in instructor.get('constraints', []):
        print(f"  - {constraint}")
    
    print("\nContext Plan:")
    for i, plan_item in enumerate(instructor.get('context_plan', [])[:3], 1):
        print(f"  {i}. Doc ID: {plan_item.get('doc_id')}")
        print(f"     Use For: {plan_item.get('use_for')}")
        print(f"     Priority: {plan_item.get('priority')}")
    
    rendered_prompt = instructor.get('rendered_prompt', '')
    print(f"\nRendered Prompt Length: {len(rendered_prompt)} characters")
    print("Note: Full passages hidden for demo brevity - only doc headers shown")
    print("\nFirst 800 characters of rendered prompt:")
    print("-" * 80)
    print(rendered_prompt[:800])
    if len(rendered_prompt) > 800:
        print("...")
    print("-" * 80)
    
    # 7. Final Grounded Answer
    print_section_header("Final Grounded Answer")
    final_answer = result.get("final_answer", {})
    
    print("\n📝 DIRECT ANSWER:")
    print("-" * 80)
    print(final_answer.get('direct_answer', 'N/A'))
    print("-" * 80)
    
    print("\n📚 EVIDENCE SUMMARY:")
    print("-" * 80)
    print(final_answer.get('evidence_summary', 'N/A'))
    print("-" * 80)
    
    print("\n⚠️  LIMITATIONS:")
    print("-" * 80)
    print(final_answer.get('limitations', 'N/A'))
    print("-" * 80)
    
    print_section_header("Test Complete")
    print("✓ All pipeline stages executed successfully!")


if __name__ == "__main__":
    # Test with default query
    test_detailed_pipeline()
    
    # Uncomment to test with a custom query:
    # test_detailed_pipeline(
    #     query="What are the treatment options for major depressive disorder?",
    #     semantic_n=8,
    #     final_k=4
    # )
