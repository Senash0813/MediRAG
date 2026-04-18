import os
import json
from statistics import mean

OUTPUT_DIR = "ablation_outputs"


def safe_len(text):
    return len(text.split()) if isinstance(text, str) and text.strip() else 0


def summarize_variant(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    ok_rows = [r for r in rows if not r.get("error")]

    avg_answer_words = mean([safe_len(r.get("answer", "")) for r in ok_rows]) if ok_rows else 0
    avg_num_docs = mean([len(r.get("retrieved_docs", [])) for r in ok_rows]) if ok_rows else 0
    gate_rejects = sum(1 for r in ok_rows if r.get("domain_gate_passed") is False)

    return {
        "num_rows": len(rows),
        "successful_rows": len(ok_rows),
        "avg_answer_words": round(avg_answer_words, 2),
        "avg_num_retrieved_docs": round(avg_num_docs, 2),
        "domain_gate_rejects": gate_rejects,
    }


def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Folder not found: {OUTPUT_DIR}")
        return

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".json"):
            path = os.path.join(OUTPUT_DIR, filename)
            print(f"\n=== {filename} ===")
            print(summarize_variant(path))


if __name__ == "__main__":
    main()