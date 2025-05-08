import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用设备: {DEVICE}")

# 加载数据
with open("question_train.json", encoding="utf-8") as f:
    train_entries = json.load(f)
with open("question_dev.json", encoding="utf-8") as f:
    dev_entries = json.load(f)
with open("question_test.json", encoding="utf-8") as f:
    test_entries = json.load(f)

# 构建模板映射
template_list = list({entry["sql"] for entry in train_entries})
template_list.append("UNK")
template2id = {tmpl: idx for idx, tmpl in enumerate(template_list)}
template2id["UNK"] = len(template_list) - 1
NUM_TEMPLATES = len(template_list)
print(f"模板总数: {NUM_TEMPLATES}")

# 构建标签映射
tags_set = {"O"}
for e in train_entries:
    for name in e.get("variables", {}).keys():
        tags_set.update({f"B-{name}", f"I-{name}"})

label2id = {lab: i for i, lab in enumerate(sorted(tags_set))}
id2label = {i: lab for lab, i in label2id.items()}

# 使用HuggingFace的分词器
TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")

def replace_ph(text: str, vars_: dict) -> str:
    for k, v in vars_.items():
        text = text.replace(k, v)
    return text

def word_tags(text: str, vars_: dict):
    words = text.split()
    tags = ["O"] * len(words)
    for var, val in vars_.items():
        tokens = val.split()
        for i in range(len(words) - len(tokens) + 1):
            if words[i:i+len(tokens)] == tokens:
                tags[i] = f"B-{var}"
                for j in range(1, len(tokens)):
                    tags[i+j] = f"I-{var}"
    return tags

def encode(texts, word_level_tags=None):
    enc = TOKENIZER(texts, truncation=True, padding=True, return_offsets_mapping=True)
    if word_level_tags is None:
        return enc, None
    aligned = []
    max_len = len(enc["input_ids"][0])
    for i, tags in enumerate(word_level_tags):
        word_ids = enc.word_ids(batch_index=i)
        seq = []
        prev = None
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            else:
                label = tags[wid] if wid < len(tags) else "O"
                if wid == prev and label.startswith("B-"):
                    label = "I-" + label[2:]
                seq.append(label2id[label])
            prev = wid
        seq += [-100] * (max_len - len(seq))
        aligned.append(seq)
    return enc, torch.tensor(aligned)

def prepare_data(entries):
    texts, temp_ids, tags = [], [], []
    for entry in entries:
        tmpl = entry["sql"]
        tid = template2id.get(tmpl, template2id["UNK"])
        txt = replace_ph(entry["text"], entry["variables"])
        texts.append(txt)
        temp_ids.append(tid)
        tags.append(word_tags(txt, entry["variables"]))
    return texts, temp_ids, tags

# 自定义模型架构
class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)  # 最大序列长度为512
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand_as(x)
        return self.embedding(x) + self.position_embedding(positions)

class SingleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_templates, num_tags):
        super().__init__()
        self.embedding = CustomEmbedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=12,
                dim_feedforward=embedding_dim * 4,
                dropout=0.1
            ),
            num_layers=12
        )
        # 槽位标注分类器
        self.tag_classifier = nn.Linear(embedding_dim, num_tags)
        # 模板分类器
        self.template_classifier = nn.Linear(embedding_dim, num_templates)
        
    def forward(self, input_ids, attention_mask=None):
        # 获取嵌入
        x = self.embedding(input_ids)
        x = x.permute(1, 0, 2)  # 转换为transformer期望的格式
        
        # 通过transformer
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        x = x.permute(1, 0, 2)  # 转换回原始格式
        
        # 槽位标注输出
        tag_logits = self.tag_classifier(x)
        
        # 模板分类输出（使用序列的最后一个有效token的表示）
        if attention_mask is not None:
            # 找到每个序列的最后一个有效token
            last_valid_idx = attention_mask.sum(dim=1) - 1
            batch_size = x.size(0)
            template_input = x[torch.arange(batch_size), last_valid_idx]
        else:
            template_input = x[:, -1]
        template_logits = self.template_classifier(template_input)
        
        return tag_logits, template_logits

class SimpleDS(Dataset):
    def __init__(self, encodings, tag_labels, template_labels):
        self.enc = encodings
        self.tag_labels = tag_labels
        self.template_labels = template_labels
        
    def __len__(self): 
        return len(self.template_labels)
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k in ["input_ids", "attention_mask"]}
        item["tag_labels"] = self.tag_labels[idx]
        item["template_labels"] = self.template_labels[idx]
        return item

# 准备数据
train_texts, train_temp_ids, train_tags = prepare_data(train_entries)
dev_texts, dev_temp_ids, dev_tags = prepare_data(dev_entries)

enc_train, tag_labels_train = encode(train_texts, train_tags)
template_labels_train = torch.tensor(train_temp_ids)

enc_dev, tag_labels_dev = encode(dev_texts, dev_tags)
template_labels_dev = torch.tensor(dev_temp_ids)

train_ds = SimpleDS(enc_train, tag_labels_train, template_labels_train)
dev_ds = SimpleDS(enc_dev, tag_labels_dev, template_labels_dev)

# 初始化模型
VOCAB_SIZE = TOKENIZER.vocab_size
EMBEDDING_DIM = 384  # 修改为384，确保能被12整除

model = SingleTransformerModel(
    VOCAB_SIZE, 
    EMBEDDING_DIM, 
    NUM_TEMPLATES, 
    len(label2id)
).to(DEVICE)

# 训练函数
def train_model(model, train_loader, dev_loader, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tag_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    template_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            tag_labels = batch["tag_labels"].to(DEVICE)
            template_labels = batch["template_labels"].to(DEVICE)
            
            optimizer.zero_grad()
            tag_logits, template_logits = model(input_ids, attention_mask)
            
            # 计算槽位标注损失
            tag_logits = tag_logits.view(-1, tag_logits.size(-1))
            tag_labels = tag_labels.view(-1)
            mask = tag_labels != -100
            tag_loss = tag_criterion(tag_logits[mask], tag_labels[mask])
            
            # 计算模板分类损失
            template_loss = template_criterion(template_logits, template_labels)
            
            # 总损失
            loss = tag_loss + template_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        model.eval()
        tag_correct = 0
        tag_total = 0
        template_correct = 0
        template_total = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                tag_labels = batch["tag_labels"].to(DEVICE)
                template_labels = batch["template_labels"].to(DEVICE)
                
                tag_logits, template_logits = model(input_ids, attention_mask)
                
                # 评估槽位标注
                tag_logits = tag_logits.view(-1, tag_logits.size(-1))
                tag_labels = tag_labels.view(-1)
                mask = tag_labels != -100
                _, predicted_tags = torch.max(tag_logits[mask], 1)
                tag_correct += (predicted_tags == tag_labels[mask]).sum().item()
                tag_total += mask.sum().item()
                
                # 评估模板分类
                _, predicted_templates = torch.max(template_logits, 1)
                template_correct += (predicted_templates == template_labels).sum().item()
                template_total += template_labels.size(0)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*template_correct/template_total:.2f}%")

# 创建数据加载器
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=16)

# 训练模型
print("\n>>> 训练模型 ...")
train_model(model, train_loader, dev_loader, num_epochs=10, learning_rate=1e-4)

# 推理和评估
print("\n>>> 开始模型推理与评估 ...")
test_texts, gold_sqls, test_tids, test_vars = [], [], [], []
missing_templates = 0

for entry in test_entries:
    tmpl = entry["sql"]
    tid = template2id.get(tmpl, template2id["UNK"])
    if tmpl not in template2id:
        missing_templates += 1

    txt = replace_ph(entry.get("text", ""), entry.get("variables", {}))
    test_texts.append(txt)
    test_tids.append(tid)
    test_vars.append(entry.get("variables", {}))

    variants = []
    sql_ = entry["sql"]
    for var in entry.get("variables", {}):
        if var in txt:
            val = entry["variables"][var]
            sql_ = sql_.replace(f'"{var}"', f'"{val}"')
    variants.append("".join(sql_.split()))
    gold_sqls.append(variants)

# 准备测试数据
enc_test = TOKENIZER(test_texts, truncation=True, padding=True, return_offsets_mapping=True, return_tensors="pt").to(DEVICE)
offsets = enc_test["offset_mapping"].cpu()

# 评估
model.eval()
with torch.no_grad():
    tag_logits, template_logits = model(enc_test["input_ids"], enc_test["attention_mask"])
    pred_tids = template_logits.argmax(dim=1).tolist()
    pred_tag_ids = tag_logits.argmax(dim=-1).tolist()

correct_tmpl, correct_full = 0, 0
errors = {
    "total_examples": len(test_texts),
    "correct_examples": 0,
    "accuracy": 0.0,
    "error_analysis": {
        "template_mismatches": 0,
        "variable_extraction_errors": 0,
        "other_errors": 0
    },
    "results": []
}

for i, txt in enumerate(test_texts):
    true_tid = test_tids[i]
    pred_tid = pred_tids[i]
    predicted_template = template_list[pred_tid] if pred_tid < len(template_list) else ""

    if pred_tid == true_tid:
        correct_tmpl += 1

    word_ids = enc_test.word_ids(batch_index=i)
    tags = [id2label[t] if t != -100 else "O" for t in pred_tag_ids[i]]

    entities, cur = [], None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            cur = None
            continue
        tag = tags[idx]
        if tag == "O":
            cur = None
            continue
        bio, name = tag.split("-", 1)
        st, ed = offsets[i][idx].tolist()
        if bio == "B" or cur is None or cur[2] != name:
            cur = [st, ed, name]
            entities.append(cur)
        else:
            cur[1] = ed
    pred_vals = {n: txt[s:e] for s, e, n in entities}

    pred_sql = predicted_template
    for var in pred_vals:
        if var in txt:
            pred_sql = pred_sql.replace(f'"{var}"', f'"{pred_vals[var]}"')
    pred_sql = "".join(pred_sql.split())

    is_correct = pred_sql in gold_sqls[i]
    if is_correct:
        correct_full += 1
    else:
        errors["results"].append({
            "text": txt,
            "is_correct": False,
            "ground_truth_sql": gold_sqls[i][0],
            "predicted_sql": pred_sql,
            "predicted_template": predicted_template
        })
        if pred_tid != true_tid:
            errors["error_analysis"]["template_mismatches"] += 1

errors["correct_examples"] = correct_full
errors["accuracy"] = correct_full / len(test_texts)

print(f"\n>>> 模板分类准确率: {correct_tmpl / len(test_texts):.3f} ({correct_tmpl}/{len(test_texts)})")
print(f">>> 整体准确率: {errors['accuracy']:.3f} ({correct_full}/{len(test_texts)+missing_templates})")
if missing_templates > 0:
    print(f">>> 出现了 {missing_templates} 个未在训练集中出现过的模板")

with open("prediction_errors.json", "w", encoding="utf-8") as f:
    json.dump(errors, f, indent=4, ensure_ascii=False)
print(">>> 错误分析保存至 prediction_errors.json")