from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QueryRewriter:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    def rewrite(self, query: str) -> str:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=64,
            num_beams=4
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
