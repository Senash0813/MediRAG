# Evaluation Folder Structure

This folder is organized by evaluation type, and each type uses a consistent layout:
- `code/` for scripts
- `data/` for datasets and intermediate files
- `results/` for output metrics/reports

## Structure

```text
evaluation/
  retrieval/
    code/
      build_retrieval_eval_dataset.py
      evaluate_retrieval_api.py
    data/
      eval_dataset_20.jsonl
      eval_dataset_20.meta.json
    results/
      aggregate_results.json
      per_query_results.csv

  generation/
    faithfulness/
      code/
        build_faithfulness_dataset.py
        build_doc_passage_dataset.py
        evaluate_faithfulness_from_passages.py
      data/
        faithfulness_dataset_20.jsonl
        faithfulness_dataset_20.meta.json
        faithfulness_with_passages_20.jsonl
        faithfulness_with_passages_20.meta.json
      results/
        faithfulness_from_passages_per_query.csv
        faithfulness_from_passages_per_query.json
        faithfulness_from_passages_summary.json

    llm_judge/
      code/
        build_answer_dataset.py
        evaluate_answers_llm_judge.py
        evaluate_reliable_only.py
      data/
        answer_eval_dataset_20.jsonl
        answer_eval_dataset_20.meta.json
        answer_eval_dataset_20_cleaned.jsonl
      results/
        answer_judge_per_query.csv
        answer_judge_per_query.json
        answer_judge_summary.json
        ragas_evaluation_reliable_only.csv

  ablation/
    code/
      convert_batch_results_to_jsonl.py
    data/
      raw_batch_results.txt
      cleaned_batch_results.jsonl
      cleaned_batch_results.report.json
      cleaned_batch_results_with_gt.jsonl
      cleaned_batch_results_with_gt.report.json
      ablation_api_response.json
    results/
      row_by_row.jsonl
      scores.jsonl
```

## Quick Run Commands

From project root:

- Build retrieval dataset:
  - `python evaluation/retrieval/code/build_retrieval_eval_dataset.py`
- Run retrieval evaluation:
  - `python evaluation/retrieval/code/evaluate_retrieval_api.py`

- Build faithfulness base dataset:
  - `python evaluation/generation/faithfulness/code/build_faithfulness_dataset.py`
- Add passage text for faithfulness:
  - `python evaluation/generation/faithfulness/code/build_doc_passage_dataset.py`
- Run faithfulness evaluation:
  - `python evaluation/generation/faithfulness/code/evaluate_faithfulness_from_passages.py --model phi:2.7b`

- Build answer-eval dataset:
  - `python evaluation/generation/llm_judge/code/build_answer_dataset.py`
- Run answer LLM-judge evaluation:
  - `python evaluation/generation/llm_judge/code/evaluate_answers_llm_judge.py --model phi:2.7b`

- Convert ablation raw output to clean JSONL:
  - `python evaluation/ablation/code/convert_batch_results_to_jsonl.py --input_file evaluation/ablation/data/raw_batch_results.txt --output_file evaluation/ablation/data/cleaned_batch_results.jsonl`

## Notes

- Retrieval uses both raw and final retrieval metrics.
- Generation evaluation is split into faithfulness and LLM-as-judge tracks.
- Ablation-specific artifacts are isolated under `evaluation/ablation`.
