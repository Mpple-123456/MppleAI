# -*- coding: utf-8 -*-
"""
MppleAI - Training Script
"""

import os
import time
import zipfile
import urllib.request
import numpy as np
from collections import Counter


class NumPyLSTM:
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.W_embed = np.random.randn(vocab_size, embed_size) * 0.1
        self.W_ih = np.random.randn(embed_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros(hidden_size)
        self.W_if = np.random.randn(embed_size, hidden_size) * 0.1
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_f = np.ones(hidden_size) * 0.5
        self.W_ic = np.random.randn(embed_size, hidden_size) * 0.1
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_c = np.zeros(hidden_size)
        self.W_io = np.random.randn(embed_size, hidden_size) * 0.1
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_o = np.zeros(hidden_size)
        self.W_out = np.random.randn(hidden_size, vocab_size) * 0.1
        self.b_out = np.zeros(vocab_size)
        self.grads = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h=None, c=None):
        batch, seq_len = x.shape
        if h is None:
            h = np.zeros((batch, self.hidden_size))
        if c is None:
            c = np.zeros((batch, self.hidden_size))
        
        outputs = np.zeros((batch, seq_len, self.vocab_size))
        states = []
        
        for t in range(seq_len):
            embed = self.W_embed[x[:, t].astype(int)]
            i = self.sigmoid(embed @ self.W_ih + h @ self.W_hh + self.b_h)
            f = self.sigmoid(embed @ self.W_if + h @ self.W_hf + self.b_f)
            g = np.tanh(embed @ self.W_ic + h @ self.W_hc + self.b_c)
            o = self.sigmoid(embed @ self.W_io + h @ self.W_ho + self.b_o)
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs[:, t, :] = h @ self.W_out + self.b_out
            states.append((h.copy(), c.copy(), embed.copy()))
        
        return outputs, states
    
    def compute_loss(self, logits, targets):
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1).astype(int)
        max_val = np.max(logits_flat, axis=-1, keepdims=True)
        probs = np.exp(logits_flat - max_val)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        correct = probs[np.arange(len(targets_flat)), targets_flat]
        loss = -np.mean(np.log(correct + 1e-8))
        return float(loss), probs.reshape(batch, seq_len, vocab_size)
    
    def backward(self, x, y, probs, states):
        batch, seq_len = x.shape
        vocab_size = self.vocab_size
        
        grad_W = {k: np.zeros_like(v) for k, v in self.__dict__.items() 
                 if k.startswith('W_') or k.startswith('b_')}
        
        dlogits = probs.copy()
        dlogits_flat = dlogits.reshape(-1, vocab_size)
        dlogits_flat[np.arange(len(y.reshape(-1))), y.reshape(-1).astype(int)] -= 1
        dlogits = dlogits_flat.reshape(batch, seq_len, vocab_size) / (batch * seq_len)
        
        dh_next = np.zeros((batch, self.hidden_size))
        dc_next = np.zeros((batch, self.hidden_size))
        
        for t in reversed(range(seq_len)):
            h, c, embed = states[t]
            prev_c = states[t-1][1] if t > 0 else np.zeros((batch, self.hidden_size))
            dh = dlogits[:, t, :] @ self.W_out.T + dh_next
            ot = self.sigmoid(embed @ self.W_io + h @ self.W_ho + self.b_o)
            ft = self.sigmoid(embed @ self.W_if + h @ self.W_hf + self.b_f)
            it = self.sigmoid(embed @ self.W_ih + h @ self.W_hh + self.b_h)
            gt = np.tanh(embed @ self.W_ic + h @ self.W_hc + self.b_c)
            do = dh * np.tanh(c) * ot * (1 - ot)
            dc = dh * ot * (1 - np.tanh(c) ** 2) + dc_next * ft
            di = dc * gt * it * (1 - it)
            df = dc * prev_c * ft * (1 - ft)
            dg = dc * it * (1 - gt ** 2)
            grad_W['W_ih'] += embed.T @ di; grad_W['W_hh'] += h.T @ di; grad_W['b_h'] += np.sum(di, axis=0)
            grad_W['W_if'] += embed.T @ df; grad_W['W_hf'] += h.T @ df; grad_W['b_f'] += np.sum(df, axis=0)
            grad_W['W_ic'] += embed.T @ dg; grad_W['W_hc'] += h.T @ dg; grad_W['b_c'] += np.sum(dg, axis=0)
            grad_W['W_io'] += embed.T @ do; grad_W['W_ho'] += h.T @ do; grad_W['b_o'] += np.sum(do, axis=0)
            grad_W['W_out'] += h.T @ dlogits[:, t, :]; grad_W['b_out'] += np.sum(dlogits[:, t, :], axis=0)
            dh_next = di @ self.W_hh.T + df @ self.W_hf.T + dg @ self.W_hc.T + do @ self.W_ho.T
            dc_next = dc * ft
        
        self.grads = grad_W
    
    def update(self, lr, clip=5.0):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in self.grads.values()))
        clip = min(1.0, clip / (total_norm + 1e-6))
        for k, g in self.grads.items():
            setattr(self, k, getattr(self, k) - lr * clip * g)
    
    def save(self, path, vocab, idx2word, epoch, loss):
        import pickle
        state = {'epoch': epoch, 'loss': loss, 'vocab': vocab, 'idx2word': idx2word}
        for k in dir(self):
            if k.startswith('W_') or k.startswith('b_'):
                state[k] = getattr(self, k)
        with open(path, 'wb') as f:
            pickle.dump(state, f)


def clean_text(text):
    return text.strip()[:200]

def build_vocab(texts, max_vocab=30000):
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
    return np.array(indices + [vocab['<PAD>']] * (max_len - len(indices)), dtype=np.int32)


def download_data():
    url = 'https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    zip_file = 'cornell_movie_dialogs_corpus.zip'
    
    print("Downloading Cornell Movie Dialogs dataset...")
    urllib.request.urlretrieve(url, zip_file)
    print("Extracting...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_file)
    print("Done!")

def load_data():
    lines_file = 'cornell movie-dialogs corpus/movie_lines.txt'
    convs_file = 'cornell movie-dialogs corpus/movie_conversations.txt'
    
    if not os.path.exists(lines_file):
        download_data()
    
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
                    src = lines.get(ids[i], '')
                    tgt = lines.get(ids[i + 1], '')
                    if src and tgt:
                        conversations.append((clean_text(src), clean_text(tgt)))
    
    return [(s, t) for s, t in conversations if s and t]


def main():
    print("=" * 50)
    print("MppleAI Training")
    print("=" * 50)
    
    print("\nLoading data...")
    conversations = load_data()
    print(f"Loaded {len(conversations)} pairs")
    
    all_text = [s for s, t in conversations] + [t for s, t in conversations]
    vocab, vocab_size = build_vocab(all_text)
    idx2word = {v: k for k, v in vocab.items()}
    print(f"Vocab: {vocab_size}")
    
    max_len = 50
    src_seqs = np.array([encode(s, vocab, max_len) for s, _ in conversations])
    tgt_seqs = np.array([encode(t, vocab, max_len) for _, t in conversations])
    
    model = NumPyLSTM(vocab_size, embed_size=64, hidden_size=128)
    print(f"Model created")
    
    epoch = 0
    best_loss = float('inf')
    batch_size = 64
    lr = 0.002
    epochs = 500
    
    print(f"\nTraining {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        start = time.time()
        
        indices = np.random.permutation(len(src_seqs))
        
        for i in range(0, len(src_seqs), batch_size):
            idx = indices[i:i + batch_size]
            src = src_seqs[idx]
            tgt = tgt_seqs[idx]
            
            logits, states = model.forward(src)
            loss, probs = model.compute_loss(logits, tgt)
            model.backward(src, tgt, probs, states)
            model.update(lr)
            
            total_loss += loss
            batches += 1
        
        avg_loss = total_loss / batches
        elapsed = time.time() - start
        
        print(f"[{epoch+1:4d}/{epochs}] Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save('chat_model.pkl', vocab, idx2word, epoch + 1, avg_loss)
            print(f"  [SAVE] Best: {best_loss:.4f}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
