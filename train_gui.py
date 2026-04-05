# -*- coding: utf-8 -*-
"""
训练 GUI - Tkinter
"""

import os
import re
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from tkinter import *
from tkinter import scrolledtext, ttk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChatLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
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


class TrainGUI:
    def __init__(self):
        self.window = Tk()
        self.window.title("MppleAI - Training")
        self.window.geometry("600x500")
        
        self.is_training = False
        self.model = None
        self.vocab = None
        self.idx2word = None
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        self.info_label = Label(self.window, text="Device: " + ("GPU" if torch.cuda.is_available() else "CPU"))
        self.info_label.pack(pady=5)
        
        self.stats_label = Label(self.window, text="Ready")
        self.stats_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(self.window, mode='indeterminate')
        self.progress.pack(fill=X, padx=20, pady=10)
        
        self.text_area = scrolledtext.ScrolledText(self.window, height=20, width=70)
        self.text_area.pack(padx=20, pady=10)
        
        btn_frame = Frame(self.window)
        btn_frame.pack(pady=10)
        
        self.start_btn = Button(btn_frame, text="Start Training", command=self.start_training, bg="green", fg="white", width=15)
        self.start_btn.pack(side=LEFT, padx=5)
        
        self.stop_btn = Button(btn_frame, text="Stop", command=self.stop_training, state=DISABLED, bg="red", fg="white", width=15)
        self.stop_btn.pack(side=LEFT, padx=5)
        
        self.loss_label = Label(self.window, text="Best Loss: --")
        self.loss_label.pack(pady=5)
    
    def load_data(self):
        self.log("Loading data...")
        conversations = load_data('data')
        if not conversations:
            self.log("No data found!")
            return
        
        conversations = [(clean_text(s), clean_text(t)) for s, t in conversations if clean_text(s) and clean_text(t)]
        self.log(f"Loaded {len(conversations)} pairs")
        
        all_text = [s for s, t in conversations] + [t for s, t in conversations]
        self.vocab, vocab_size = build_vocab(all_text)
        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.log(f"Vocab: {vocab_size} words")
        
        max_len = 25
        src_seqs = np.array([encode(s, self.vocab, max_len) for s, _ in conversations])
        tgt_seqs = np.array([encode(t, self.vocab, max_len) for _, t in conversations])
        
        self.loader = DataLoader(src_seqs, tgt_seqs, batch_size=512)
        self.log(f"DataLoader ready: {self.loader.n // self.loader.batch_size} batches")
        
        self.model = ChatLSTM(vocab_size, embed_size=256, hidden_size=512, num_layers=2).to(device)
        params = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model: {params/1e6:.1f}M parameters")
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        
        self.log("Ready to train!")
    
    def log(self, msg):
        self.text_area.insert(END, msg + "\n")
        self.text_area.see(END)
        self.window.update()
    
    def start_training(self):
        if self.is_training:
            return
        
        if self.model is None:
            self.log("Error: Model not loaded!")
            return
        
        self.is_training = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.progress.start()
        
        self.train_thread = threading.Thread(target=self.train_loop)
        self.train_thread.daemon = True
        self.train_thread.start()
    
    def stop_training(self):
        self.is_training = False
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.progress.stop()
        self.log("Training stopped.")
    
    def train_loop(self):
        epoch = 0
        best_loss = float('inf')
        model_path = 'chat_model.pkl'
        
        while self.is_training:
            epoch += 1
            self.model.train()
            total_loss = 0
            batches = 0
            start = time.time()
            
            for src, tgt in self.loader:
                self.optimizer.zero_grad()
                logits, _ = self.model(src)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]['lr']
            
            self.stats_label.config(text=f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {lr:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.loss_label.config(text=f"Best Loss: {best_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'model_state': self.model.state_dict(),
                    'vocab': self.vocab,
                    'idx2word': self.idx2word,
                }, model_path)
                self.log(f"[SAVE] Epoch {epoch} | Loss: {avg_loss:.4f}")
            
            if epoch % 10 == 0:
                self.log(f"[CHECK] Epoch {epoch} | Best: {best_loss:.4f}")
        
        self.progress.stop()
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = TrainGUI()
    gui.run()
