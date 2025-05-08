# -*- coding: utf-8 -*-
"""feedforward_classification.py - 使用前馈神经网络进行SQL模板分类和槽位标注"""

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

# 前馈神经网络模型
class FeedforwardEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

class FeedforwardClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_labels, hidden_dims=[512, 256]):
        super().__init__()
        self.embedding = FeedforwardEmbedding(vocab_size, embedding_dim)
        
        # 构建多层前馈网络
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feedforward = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        # 获取词嵌入
        x = self.embedding(input_ids)
        
        # 使用attention mask进行平均池化
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
            
        # 通过前馈网络
        x = self.feedforward(x)
        return self.classifier(x)

class FeedforwardTokenClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_labels, hidden_dims=[512, 256]):
        super().__init__()
        self.embedding = FeedforwardEmbedding(vocab_size, embedding_dim)
        
        # 构建多层前馈网络
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feedforward = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        # 获取词嵌入
        x = self.embedding(input_ids)
        
        # 对每个token单独处理
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
        
        # 通过前馈网络
        x = self.feedforward(x)
        x = self.classifier(x)
        
        # 重塑回原始维度
        return x.view(batch_size, seq_len, -1)

class SimpleDS(Dataset):
    def __init__(self, encodings, labels):
        self.enc, self.labels = encodings, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k in ["input_ids", "attention_mask"]}
        item["labels"] = self.labels[idx]
        return item

# 准备数据
train_texts, train_temp_ids, train_tags = prepare_data(train_entries)
dev_texts, dev_temp_ids, dev_tags = prepare_data(dev_entries)

enc_cls_train = TOKENIZER(train_texts, truncation=True, padding=True)
cls_labels_train = torch.tensor(train_temp_ids)
enc_cls_dev = TOKENIZER(dev_texts, truncation=True, padding=True)
cls_labels_dev = torch.tensor(dev_temp_ids)

enc_tok_train, tok_labels_train = encode(train_texts, train_tags)
enc_tok_dev, tok_labels_dev = encode(dev_texts, dev_tags)

train_ds_cls = SimpleDS(enc_cls_train, cls_labels_train)
dev_ds_cls = SimpleDS(enc_cls_dev, cls_labels_dev)
train_ds_tok = SimpleDS(enc_tok_train, tok_labels_train)
dev_ds_tok = SimpleDS(enc_tok_dev, tok_labels_dev)

# 初始化模型
VOCAB_SIZE = TOKENIZER.vocab_size
EMBEDDING_DIM = 256

cls_model = FeedforwardClassificationModel(VOCAB_SIZE, EMBEDDING_DIM, NUM_TEMPLATES).to(DEVICE)
tok_model = FeedforwardTokenClassificationModel(VOCAB_SIZE, EMBEDDING_DIM, len(label2id)).to(DEVICE)

# 训练函数
def train_model(model, train_loader, dev_loader, num_epochs, learning_rate, is_token_classification=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) if is_token_classification else nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            if is_token_classification:
                # 对于token分类，我们需要重塑输出和标签
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                # 只计算非填充位置的损失
                mask = labels != -100
                outputs = outputs[mask]
                labels = labels[mask]
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                
                if is_token_classification:
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                    mask = labels != -100
                    outputs = outputs[mask]
                    labels = labels[mask]
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# 创建数据加载器
train_loader_cls = DataLoader(train_ds_cls, batch_size=16, shuffle=True)
dev_loader_cls = DataLoader(dev_ds_cls, batch_size=16)
train_loader_tok = DataLoader(train_ds_tok, batch_size=16, shuffle=True)
dev_loader_tok = DataLoader(dev_ds_tok, batch_size=16)

# 训练模型
print("\n>>> 训练模板分类器 ...")
train_model(cls_model, train_loader_cls, dev_loader_cls, num_epochs=20, learning_rate=1e-4, is_token_classification=False)

print("\n>>> 训练槽位标注器 ...")
train_model(tok_model, train_loader_tok, dev_loader_tok, num_epochs=7, learning_rate=1e-4, is_token_classification=True)

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
cls_model.eval()
tok_model.eval()
with torch.no_grad():
    logits_cls = cls_model(enc_test["input_ids"], enc_test["attention_mask"])
    pred_tids = logits_cls.argmax(dim=1).tolist()

    logits_tok = tok_model(enc_test["input_ids"], enc_test["attention_mask"])
    pred_tag_ids = logits_tok.argmax(dim=-1).tolist()

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

# with open("prediction_errors.json", "w", encoding="utf-8") as f:
#     json.dump(errors, f, indent=4, ensure_ascii=False)
# print(">>> 错误分析保存至 prediction_errors.json") 