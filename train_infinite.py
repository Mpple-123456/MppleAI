# -*- coding: utf-8 -*-
"""
无限训练脚本 - GPU 加速
"""

import os
import re
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 50)
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)


class ChatLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embed = self.dropout(self.embedding(x))
        if hidden is None:
            out, hidden = self.lstm(embed)
        else:
            out, hidden = self.lstm(embed, hidden)
        return self.fc(out), hidden


def clean_text(text):
    return text.strip()[:200]

def build_vocab(texts, max_vocab=50000):
    counter = Counter()
    for text in texts:
        for char in text:
            if char.strip():
                counter[char] += 1
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for char, _ in counter.most_common(max_vocab - 4):
        vocab[char] = len(vocab)
    return vocab, len(vocab)

def encode(sentence, vocab, max_len=50):
    chars = list(sentence)[:max_len-2]
    indices = [vocab.get(c, vocab['<UNK>']) for c in chars]
    indices = [vocab['<SOS>']] + indices + [vocab['<EOS>']]
    return np.array(indices + [vocab['<PAD>']] * (max_len - len(indices)), dtype=np.int64)


class DataLoader:
    def __init__(self, src, tgt, batch_size):
        self.src = torch.LongTensor(src)
        self.tgt = torch.LongTensor(tgt)
        self.batch_size = batch_size
        self.n = len(src)
    
    def __iter__(self):
        indices = torch.randperm(self.n)
        for i in range(0, self.n, self.batch_size):
            idx = indices[i:i + self.batch_size]
            yield self.src[idx].to(device), self.tgt[idx].to(device)


def load_data(data_dir):
    lines_file = os.path.join(data_dir, 'movie_lines.txt')
    convs_file = os.path.join(data_dir, 'movie_conversations.txt')
    
    if not os.path.exists(lines_file):
        return None
    
    lines = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 5:
                lines[parts[0]] = parts[4].strip()
    
    conversations = []
    with open(convs_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 4:
                ids = eval(parts[3])
                for i in range(len(ids) - 1):
                    src, tgt = lines.get(ids[i], ''), lines.get(ids[i+1], '')
                    if src and tgt:
                        conversations.append((src, tgt))
    
    return conversations


def infinite_train():
    model_path = 'chat_model.pkl'
    data_dir = 'data'
    
    print("\n[LOAD] Loading data...")
    conversations = load_data(data_dir)
    if not conversations:
        print("[ERROR] No data found!")
        return
    
    conversations = [(clean_text(s), clean_text(t)) for s, t in conversations 
                    if clean_text(s) and clean_text(t)]
    print(f"[DATA] {len(conversations)} pairs")
    
    print("[VOCAB] Building...")
    all_text = [s for s, t in conversations] + [t for s, t in conversations]
    vocab, vocab_size = build_vocab(all_text)
    idx2word = {v: k for k, v in vocab.items()}
    print(f"[VOCAB] {vocab_size} words")
    
    print("[ENCODE] Encoding...")
    max_len = 25
    src_seqs = np.array([encode(s, vocab, max_len) for s, _ in conversations])
    tgt_seqs = np.array([encode(t, vocab, max_len) for _, t in conversations])
    
    batch_size = 256
    loader = DataLoader(src_seqs, tgt_seqs, batch_size)
    
    print("[MODEL] Creating...")
    model = ChatLSTM(vocab_size, embed_size=128, hidden_size=256, num_layers=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {params / 1e6:.1f}M parameters")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    start_epoch = 0
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        vocab = checkpoint['vocab']
        idx2word = checkpoint['idx2word']
        start_epoch = checkpoint['epoch']
        print(f"[LOAD] Resuming epoch {start_epoch}, loss={checkpoint['loss']:.4f}")
    
    print("\n" + "=" * 50)
    print("INFINITE TRAINING - Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    epoch = start_epoch
    best_loss = float('inf')
    
    try:
        while True:
            epoch += 1
            model.train()
            total_loss = 0
            batches = 0
            start = time.time()
            
            for src, tgt in loader:
                optimizer.zero_grad()
                logits, _ = model(src)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]['lr']
            
            print(f"[{epoch:05d}] Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {lr:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'model_state': model.state_dict(),
                    'vocab': vocab,
                    'idx2word': idx2word,
                }, model_path)
                print(f"  [SAVE] Best: {best_loss:.4f}")
            
            if epoch % 50 == 0:
                print(f"[CHECKPOINT] Best loss: {best_loss:.4f}")
    
    except KeyboardInterrupt:
        print("\n\n[TRAIN] Stopped. Model saved.")
        torch.save({
            'epoch': epoch,
            'loss': avg_loss,
            'model_state': model.state_dict(),
            'vocab': vocab,
            'idx2word': idx2word,
        }, model_path)


if __name__ == "__main__":
    infinite_train()
