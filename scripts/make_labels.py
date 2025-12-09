from pathlib import Path
import json

train_root = Path("data/gtsrb/train")

if not train_root.exists():
    print("ERROR: data/gtsrb/train not found. Update path if needed.")
    exit()

classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])

labels = {str(i): name for i, name in enumerate(classes)}

out = Path("data/gtsrb/labels.json")
out.parent.mkdir(parents=True, exist_ok=True)

out.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

print("Wrote", out)
print("First 10 entries:")
for k in list(sorted(labels.keys(), key=int))[:10]:
    print(k, "->", labels[k])
