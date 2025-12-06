from datasets import load_dataset

ds = load_dataset("miriad/miriad-4.4M", split="train", streaming=True)

unique_values = set()
for row in ds:
    unique_values.add(row["specialty"])

print(unique_values)
