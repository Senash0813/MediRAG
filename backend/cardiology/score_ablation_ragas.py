import os
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, answer_relevancy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# ============================================================
# LOAD ENV
# ============================================================
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = "ablation_outputs"
RESULTS_DIR = "ablation_scores"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD VARIANT DATA
# ============================================================
def load_variant_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    cleaned = []
    for row in rows:
        if row.get("error"):
            continue

        question = row.get("query", "").strip()
        answer = row.get("answer", "").strip()
        ground_truth = row.get("ground_truth", "").strip()
        contexts = row.get("retrieved_docs", [])

        if not isinstance(contexts, list):
            contexts = []

        if question and answer and ground_truth and contexts:
            cleaned.append(
                {
                    # ragas>=0.4 expects these column names by default
                    "user_input": question,
                    "response": answer,
                    "reference": ground_truth,
                    "retrieved_contexts": contexts,
                }
            )

    return cleaned


# ============================================================
# SAFE RESULT CONVERTER
# ============================================================
def evaluation_result_to_dict(result) -> Dict[str, Any]:
    try:
        return dict(result)
    except Exception:
        pass

    scores = {}
    for key in ["answer_correctness", "faithfulness", "answer_relevancy"]:
        try:
            scores[key] = result[key]
        except Exception:
            scores[key] = None
    return scores


# ============================================================
# SCORE SINGLE VARIANT
# ============================================================
def score_variant(variant_filename: str) -> Dict[str, Any]:
    variant_path = os.path.join(OUTPUT_DIR, variant_filename)
    rows = load_variant_json(variant_path)

    if not rows:
        raise ValueError(f"No valid rows found in {variant_filename}")

    dataset = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=120)
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            answer_correctness,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    scores = evaluation_result_to_dict(result)

    out_path = os.path.join(
        RESULTS_DIR,
        variant_filename.replace(".json", "_ragas_scores.json")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    return scores


# ============================================================
# MAIN
# ============================================================
def main():
    variant_files = [
        "baseline_rag.json",
        "hyde_only.json",
        "projection_only.json",
        "fusion_no_domain_gate.json",
        "full_cardiorag.json",
    ]

    summary_rows = []

    for filename in variant_files:
        path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(path):
            print(f"Skipping missing file: {filename}")
            continue

        print(f"\nScoring {filename} ...")
        scores = score_variant(filename)

        summary_rows.append({
            "variant": filename.replace(".json", ""),
            "answer_correctness": scores.get("answer_correctness"),
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
        })

        print("Scores:", scores)

    summary_path = os.path.join(RESULTS_DIR, "ragas_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()