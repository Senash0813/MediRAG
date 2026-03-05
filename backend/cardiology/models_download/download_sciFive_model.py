from transformers import AutoModel, AutoTokenizer

model_name = "SandunR/hyde-sciFive-cardiology-generator"
local_dir = "models/hyde-sciFive-cardiology-generator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print(f"Model and tokenizer saved to {local_dir}")