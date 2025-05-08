# -*- coding: utf-8 -*-
"""generation.py - 使用自定义seq2seq模型进行SQL生成"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用设备: {DEVICE}")

# 替换占位符为变量真实值
def replace_variables(text, sql, variables):
    text_placeholders = set()
    for var in variables.keys():
        if var in text:
            text_placeholders.add(var)
    
    for var in text_placeholders:
        val = variables[var]
        text = text.replace(var, val)
        sql = sql.replace(f"\"{var}\"", f"\"{val}\"")
    return text, sql

# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        text, sql = replace_variables(item["text"], item["sql"], item["variables"])
        data.append({"input": text, "output": sql})
    return data

# 自定义数据集类
class SQLDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_len=128, target_max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = self.tokenizer(
            item["input"],
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer(
            item["output"],
            max_length=self.target_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze()
        }

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 自定义Seq2Seq模型
class CustomSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # 处理源序列的attention mask
        if src_mask is not None:
            src_mask = src_mask.bool()
            # 创建正确的attention mask形状 [batch_size * num_heads, seq_len, seq_len]
            batch_size = src.size(0)
            seq_len = src.size(1)
            num_heads = self.encoder.layers[0].self_attn.num_heads
            
            # 首先将mask扩展到序列长度维度
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            src_mask = src_mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
            
            # 然后扩展到注意力头维度
            src_mask = src_mask.expand(batch_size, num_heads, seq_len, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            src_mask = src_mask.reshape(-1, seq_len, seq_len)  # [batch_size * num_heads, seq_len, seq_len]
            src_mask = (~src_mask) * torch.finfo(src.dtype).min
            
        memory = self.encoder(src, src_mask)
        
        # 解码器
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt, memory, tgt_mask)
        
        return self.fc(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, input_ids, max_length=256, num_beams=2):
        self.eval()
        with torch.no_grad():
            # 编码
            src = self.embedding(input_ids)
            src = self.pos_encoder(src)
            
            # 处理源序列的attention mask
            batch_size = src.size(0)
            seq_len = src.size(1)
            num_heads = self.encoder.layers[0].self_attn.num_heads
            
            # 创建全1的attention mask
            src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            src_mask = src_mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
            src_mask = src_mask.expand(batch_size, num_heads, seq_len, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            src_mask = src_mask.reshape(-1, seq_len, seq_len)  # [batch_size * num_heads, seq_len, seq_len]
            src_mask = (~src_mask) * torch.finfo(src.dtype).min
            
            memory = self.encoder(src, src_mask)
            
            # 初始化目标序列，使用开始标记
            tgt = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)  # 使用1作为开始标记
            
            # 自回归生成
            for _ in range(max_length):
                tgt_emb = self.embedding(tgt)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
                out = self.decoder(tgt_emb, memory, tgt_mask)
                out = self.fc(out[:, -1:])
                
                # beam search
                if num_beams > 1:
                    probs = torch.softmax(out, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, num_beams, dim=-1)
                    next_token = topk_indices[:, 0, 0].unsqueeze(-1)
                else:
                    next_token = torch.argmax(out, dim=-1)
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否生成了结束标记（假设2是结束标记）
                if (next_token == 2).any():
                    break
                    
            return tgt

# 训练函数
def train_model(model, train_loader, dev_loader, num_epochs=5, learning_rate=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # 准备目标序列
            tgt_input = labels[:, :-1]  # 去掉最后一个token
            tgt_output = labels[:, 1:]  # 去掉第一个token
            
            optimizer.zero_grad()
            outputs = model(input_ids, tgt_input, attention_mask)
            
            # 确保输出和目标的维度匹配
            outputs = outputs.reshape(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
            tgt_output = tgt_output.reshape(-1)  # [batch_size * seq_len]
            
            loss = criterion(outputs, tgt_output)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                
                outputs = model(input_ids, tgt_input, attention_mask)
                
                # 确保输出和目标的维度匹配
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(outputs, tgt_output)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(dev_loader):.4f}")

# 评估函数
def evaluate_model(model, tokenizer, test_data, max_length=256):
    model.eval()
    correct = 0
    total = 0
    exact_match = 0  # 完全匹配的样本数
    partial_match = 0  # 部分匹配的样本数（包含关键SQL元素）
    
    print("\n开始评估模型性能...")
    print("=" * 50)
    
    for item in tqdm(test_data, desc="Evaluating"):
        input_text = item["input"]
        target_sql = item["output"]
        
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).input_ids.to(DEVICE)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length)
        
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 打印前5个示例
        if total < 5:
            print(f"\n示例 {total + 1}:")
            print(f"输入: {input_text}")
            print(f"目标: {target_sql}")
            print(f"预测: {pred}")
            print("-" * 50)
        
        # 检查完全匹配
        if pred == target_sql:
            exact_match += 1
            correct += 1
        
        # 检查部分匹配（包含关键SQL元素）
        target_elements = set(target_sql.lower().split())
        pred_elements = set(pred.lower().split())
        common_elements = target_elements.intersection(pred_elements)
        if len(common_elements) / len(target_elements) > 0.5:  # 如果超过50%的关键元素匹配
            partial_match += 1
        
        total += 1
        
        if total % 10 == 0:
            print(f"\n当前进度: {total}/{len(test_data)}")
            print(f"完全匹配准确率: {exact_match/total:.4f}")
            print(f"部分匹配准确率: {partial_match/total:.4f}")
    
    # 计算最终准确率
    exact_accuracy = exact_match / total
    partial_accuracy = partial_match / total
    
    print("\n" + "=" * 50)
    print("模型评估结果:")
    print(f"总样本数: {total}")
    print(f"完全匹配样本数: {exact_match}")
    print(f"部分匹配样本数: {partial_match}")
    print(f"完全匹配准确率: {exact_accuracy:.4f}")
    print(f"部分匹配准确率: {partial_accuracy:.4f}")
    print("=" * 50)
    
    return exact_accuracy, partial_accuracy

def main():
    # 加载数据
    print("加载数据...")
    train_data = load_data("question_train.json")
    dev_data = load_data("question_dev.json")
    test_data = load_data("question_test.json")
    
    # 初始化分词器
    print("初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # 只使用分词器
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = SQLDataset(train_data, tokenizer)
    dev_dataset = SQLDataset(dev_data, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)
    
    # 初始化模型
    print("初始化模型...")
    model = CustomSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(DEVICE)
    
    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, dev_loader)
    
    # 评估模型
    print("开始评估...")
    exact_acc, partial_acc = evaluate_model(model, tokenizer, test_data)
    
    # 保存模型
    print("保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'exact_accuracy': exact_acc,
        'partial_accuracy': partial_acc
    }, "custom_seq2seq.pth")
    
    print(f"\n模型已保存，完全匹配准确率: {exact_acc:.4f}, 部分匹配准确率: {partial_acc:.4f}")

if __name__ == "__main__":
    main()
