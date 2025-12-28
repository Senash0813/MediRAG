from transformers import AutoModel, AutoTokenizer

model_name = "KomeijiForce/inbedder-roberta-large"
local_dir = "models/inbedder-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print(f"Model and tokenizer saved to {local_dir}")