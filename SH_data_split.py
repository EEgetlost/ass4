import json
from pathlib import Path
def save_json(data, path):
    """保存JSON数据到指定路径"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 加载数据
data_path = Path(__file__).parent / "atis.json"
assert data_path.exists(), "请将 atis.json 与脚本置于同一目录"
with data_path.open(encoding="utf-8") as f:
    data = json.load(f)



train_entries, dev_entries, test_entries = [], [], []

for entry in data:
    for sent in entry.get("sentences", []):
        item = {
            "text": sent.get("text", ""),
            "variables": sent.get("variables", {}),
            "sql": min(entry["sql"], key=len),
        }
        split = sent.get("question-split")
        if split == "train":
            train_entries.append(item)
        elif split == "dev":
            dev_entries.append(item)
        elif split == "test":
            test_entries.append(item)
save_json(train_entries, "question_train.json")
save_json(dev_entries, "question_dev.json")
save_json(test_entries, "question_test.json")

# for entry in data:
#     split = entry.get("query-split")
#     for sent in entry.get("sentences", []):
#         item = {
#             "text": sent.get("text", ""),
#             "variables": sent.get("variables", {}),
#             "sql": min(entry["sql"], key=len),
#         }
#         if split == "train":
#             train_entries.append(item)
#         elif split == "dev":
#             dev_entries.append(item)
#         elif split == "test":
#             test_entries.append(item)
# save_json(train_entries, "query_train.json")
# save_json(dev_entries, "query_dev.json")
# save_json(test_entries, "query_test.json")
