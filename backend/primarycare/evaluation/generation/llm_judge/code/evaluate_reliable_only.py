"""
Simplified evaluation - only use metrics that work reliably
(answer_relevancy and answer_similarity - both work without timeouts)
"""
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_similarity
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("="*80)
print("SIMPLIFIED RAGAS EVALUATION")
print("Metrics: answer_relevancy, answer_similarity (no faithfulness)")
print("="*80)

# Load data
with open('combined_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', data.get('combined_results', []))

# Prepare dataset
eval_data = {
    'question': [r['question'] for r in results],
    'answer': [r.get('answer', r.get('pipeline_response', '')) for r in results],
    'ground_truth': [r['ground_truth'] for r in results],
}

dataset = Dataset.from_dict(eval_data)

print(f"\nDataset: {len(dataset)} questions\n")

# Setup LLM and embeddings
print("Setting up LLM and embeddings...")
llm = ChatOpenAI(
    model="local-model",
    openai_api_key=LM_STUDIO_API_KEY,
    openai_api_base=LM_STUDIO_BASE_URL,
    temperature=0.1,
    max_tokens=2048,
    request_timeout=180
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("✅ Setup complete\n")

# Run evaluation (only 2 metrics that don't timeout)
print("Starting evaluation...")
print("Metrics: answer_relevancy, answer_similarity\n")

result = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy, answer_similarity],
    llm=llm,
    embeddings=embeddings
)

print("\n" + "="*80)
print("EVALUATION RESULTS (2 RELIABLE METRICS)")
print("="*80)

# Convert to pandas and save
df = result.to_pandas()
out_path = Path("evaluation/generation/llm_judge/results/ragas_evaluation_reliable_only.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

# Display results
print("\n📊 OVERALL SCORES:")
print("-"*80)

scores = {}
for metric in ['answer_relevancy', 'answer_similarity']:
    if metric in df.columns:
        valid = df[metric].dropna()
        if len(valid) > 0:
            mean = valid.mean()
            scores[metric] = mean
            failed = len(df) - len(valid)
            status = f"  ⚠️ ({failed}/{len(df)} failed)" if failed > 0 else ""
            print(f"{metric:30s}: {mean:.4f}{status}")

if scores:
    overall = sum(scores.values()) / len(scores)
    print("-"*80)
    print(f"{'OVERALL SCORE (2 metrics)':30s}: {overall:.4f}")
    print("="*80)
    
    print("\n💡 INTERPRETATION:")
    if overall >= 0.85:
        print("   🟢 EXCELLENT")
    elif overall >= 0.75:
        print("   🟡 GOOD")
    elif overall >= 0.65:
        print("   🟠 FAIR")
    else:
        print("   🔴 NEEDS WORK")

print(f"\n✅ Results saved to: {out_path}")
print("="*80)
