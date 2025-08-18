import re
import json

# 读取 README 文件
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# 去掉 Other Resources 之后的部分
content = re.split(r'## 🌐 Other Resources', content)[0]

# 匹配 Markdown 表格的每行第一列数据集名称
datasets = []
for line in content.splitlines():
    # 跳过表头和分隔线
    if line.strip().startswith("|") and not re.match(r"\|\s*-+", line):
        cols = line.split("|")
        if len(cols) > 1:
            name = cols[1].strip()
            # 去掉 []() 链接
            name = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", name)
            if name:
                datasets.append(name)

# 去重
datasets = list(dict.fromkeys(datasets))
dataset_count = len(datasets)

# 输出到 JSON，用于 Shields.io 徽章
badge_json = {
    "schemaVersion": 1,
    "label": "Total Datasets",
    "message": str(dataset_count),
    "color": "blue"
}

with open("scripts/dataset_count.json", "w", encoding="utf-8") as f:
    json.dump(badge_json, f, ensure_ascii=False, indent=2)