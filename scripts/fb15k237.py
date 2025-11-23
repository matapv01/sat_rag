import os
import json
from datasets import load_dataset

DATA_DIR = "data/fb15k237"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading FB15k-237 from Hugging Face...")
ds = load_dataset("KGraph/FB15k-237")
print("Dataset loaded!")
print("Available splits:", ds.keys())

# Chọn split tồn tại
splits = [s for s in ["train", "valid", "test"] if s in ds.keys()]

# Xuất file txt
for split in splits:
    fname = os.path.join(DATA_DIR, f"{split}.txt")
    with open(fname, "w") as f:
        for row in ds[split]:
            h,r,t = row['text'].split('\t')
            f.write(f"{h} {r} {t}\n")
    print(f"{split}.txt saved!")

# Tạo entity2text.json dummy
entities = set()
for split in splits:
    for row in ds[split]:
        h,r,t = row['text'].split('\t')
        entities.add(h)
        entities.add(t)

entity2text = {e: f"Description for entity {e}" for e in entities}
with open(os.path.join(DATA_DIR, "entity2text.json"), "w") as f:
    json.dump(entity2text, f, indent=2)

print("entity2text.json created!")
print(f"FB15k-237 ready at {DATA_DIR}")
