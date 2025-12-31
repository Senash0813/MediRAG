from __future__ import annotations

from medirag.config import load_settings
from medirag.pipeline.orchestrator import init_assets, run_pipeline


def main():
    settings = load_settings()
    assets = init_assets(settings)

    query = "What is HIV?"
    result = run_pipeline(assets=assets, query=query, k=5)

    print("=== MODEL ANSWER ===")
    print(result["answer_text"])


if __name__ == "__main__":
    main()
