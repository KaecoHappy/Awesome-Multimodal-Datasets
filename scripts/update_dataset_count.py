import re
import json

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

dataset_count = len(re.findall(r"^\s*[-*]\s+.+", content, re.MULTILINE))

data = {
    "schemaVersion": 1,
    "label": "datasets",
    "message": str(dataset_count),
    "color": "blue"
}

with open("dataset_count.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
