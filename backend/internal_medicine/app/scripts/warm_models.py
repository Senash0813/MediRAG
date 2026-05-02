from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


def _load_sentence_transformer(model_name: str) -> None:
    print(f"Warming SentenceTransformer model: {model_name}")
    SentenceTransformer(model_name)


def _load_pipeline(task: str, model_name: str, device: int) -> None:
    from transformers import pipeline

    print(f"Warming Transformers {task} model: {model_name}")
    pipeline(task, model=model_name, device=device)


def main() -> None:
    settings = get_settings()

    _load_sentence_transformer(settings.embedding_model_name)

    if settings.enable_ner:
        _load_pipeline(
            "ner",
            settings.transformer_ner_model,
            settings.transformers_device,
        )

    if settings.enable_nli:
        _load_pipeline(
            "text-classification",
            settings.nli_model,
            settings.transformers_device,
        )

    if str(settings.judge_backend).strip().lower() == "hf":
        _load_pipeline(
            "text2text-generation",
            settings.hf_judge_model,
            settings.transformers_device,
        )

    print("Model warmup complete.")


if __name__ == "__main__":
    main()