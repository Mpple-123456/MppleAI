# -*- coding: utf-8 -*-
"""
MppleAI - Pure NumPy LSTM Chat
No PyTorch Required!
"""

import os
import re
import time
import threading
import numpy as np
from collections import Counter
from tkinter import *
from tkinter import scrolledtext, filedialog, messagebox


class NumPyLSTM:
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        scale = 0.1
        self.W_embed = np.random.randn(vocab_size, embed_size) * scale
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
        
        grad_W_embed = np.zeros_like(self.W_embed)
        grad_W_ih = np.zeros_like(self.W_ih)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_b_h = np.zeros_like(self.b_h)
        grad_W_if = np.zeros_like(self.W_if)
        grad_W_hf = np.zeros_like(self.W_hf)
        grad_b_f = np.zeros_like(self.b_f)
        grad_W_ic = np.zeros_like(self.W_ic)
        grad_W_hc = np.zeros_like(self.W_hc)
        grad_b_c = np.zeros_like(self.b_c)
        grad_W_io = np.zeros_like(self.W_io)
        grad_W_ho = np.zeros_like(self.W_ho)
        grad_b_o = np.zeros_like(self.b_o)
        grad_W_out = np.zeros_like(self.W_out)
        grad_b_out = np.zeros_like(self.b_out)
        
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
            
            grad_W_ih += embed.T @ di
            grad_W_hh += h.T @ di
            grad_b_h += np.sum(di, axis=0)
            
            grad_W_if += embed.T @ df
            grad_W_hf += h.T @ df
            grad_b_f += np.sum(df, axis=0)
            
            grad_W_ic += embed.T @ dg
            grad_W_hc += h.T @ dg
            grad_b_c += np.sum(dg, axis=0)
            
            grad_W_io += embed.T @ do
            grad_W_ho += h.T @ do
            grad_b_o += np.sum(do, axis=0)
            
            grad_W_out += h.T @ dlogits[:, t, :]
            grad_b_out += np.sum(dlogits[:, t, :], axis=0)
            
            indices = x[:, t].astype(int)
            for b in range(batch):
                grad_W_embed[indices[b]] += di[b] @ self.W_ih.T + df[b] @ self.W_if.T + dg[b] @ self.W_ic.T + do[b] @ self.W_io.T
            
            dh_next = di @ self.W_hh.T + df @ self.W_hf.T + dg @ self.W_hc.T + do @ self.W_ho.T
            dc_next = dc * ft
        
        self.grads = {
            'W_embed': grad_W_embed, 'W_ih': grad_W_ih, 'W_hh': grad_W_hh, 'b_h': grad_b_h,
            'W_if': grad_W_if, 'W_hf': grad_W_hf, 'b_f': grad_b_f,
            'W_ic': grad_W_ic, 'W_hc': grad_W_hc, 'b_c': grad_b_c,
            'W_io': grad_W_io, 'W_ho': grad_W_ho, 'b_o': grad_b_o,
            'W_out': grad_W_out, 'b_out': grad_b_out,
        }
    
    def update(self, lr, clip=5.0):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in self.grads.values()))
        clip = min(1.0, clip / (total_norm + 1e-6))
        
        self.W_embed -= lr * clip * self.grads['W_embed']
        self.W_ih -= lr * clip * self.grads['W_ih']
        self.W_hh -= lr * clip * self.grads['W_hh']
        self.b_h -= lr * clip * self.grads['b_h']
        self.W_if -= lr * clip * self.grads['W_if']
        self.W_hf -= lr * clip * self.grads['W_hf']
        self.b_f -= lr * clip * self.grads['b_f']
        self.W_ic -= lr * clip * self.grads['W_ic']
        self.W_hc -= lr * clip * self.grads['W_hc']
        self.b_c -= lr * clip * self.grads['b_c']
        self.W_io -= lr * clip * self.grads['W_io']
        self.W_ho -= lr * clip * self.grads['W_ho']
        self.b_o -= lr * clip * self.grads['b_o']
        self.W_out -= lr * clip * self.grads['W_out']
        self.b_out -= lr * clip * self.grads['b_out']
    
    def save(self, path, vocab, idx2word, epoch, loss):
        state = {
            'epoch': epoch, 'loss': loss,
            'vocab_size': self.vocab_size, 'embed_size': self.embed_size, 'hidden_size': self.hidden_size,
            'W_embed': self.W_embed, 'W_ih': self.W_ih, 'W_hh': self.W_hh, 'b_h': self.b_h,
            'W_if': self.W_if, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_ic': self.W_ic, 'W_hc': self.W_hc, 'b_c': self.b_c,
            'W_io': self.W_io, 'W_ho': self.W_ho, 'b_o': self.b_o,
            'W_out': self.W_out, 'b_out': self.b_out,
            'vocab': vocab, 'idx2word': idx2word,
        }
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.vocab_size = state['vocab_size']
        self.embed_size = state['embed_size']
        self.hidden_size = state['hidden_size']
        self.W_embed = state['W_embed']
        self.W_ih = state['W_ih']
        self.W_hh = state['W_hh']
        self.b_h = state['b_h']
        self.W_if = state['W_if']
        self.W_hf = state['W_hf']
        self.b_f = state['b_f']
        self.W_ic = state['W_ic']
        self.W_hc = state['W_hc']
        self.b_c = state['b_c']
        self.W_io = state['W_io']
        self.W_ho = state['W_ho']
        self.b_o = state['b_o']
        self.W_out = state['W_out']
        self.b_out = state['b_out']
        
        return state['vocab'], state['idx2word'], state['epoch'], state['loss']


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

def generate(model, text, vocab, idx2word, max_len=50, temperature=0.15):
    src = encode(text, vocab, max_len).reshape(1, -1)
    h = c = None
    result = []
    eos = vocab['<EOS>']
    pad = vocab['<PAD>']
    
    for _ in range(max_len):
        logits, states = model.forward(src, h, c)
        h, c, _ = states[-1]
        
        probs = np.exp(logits[0, -1] / temperature)
        probs[pad] = 0
        probs = probs / probs.sum()
        token = np.argmax(probs)
        
        if token == eos:
            break
        if token not in [0, 1, 2, 3]:
            word = idx2word.get(token, '')
            if word:
                result.append(word)
        
        src = np.append(src[0, 1:], token).reshape(1, -1)
    
    return ''.join(result) or "i am not sure"


class MppleAI:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.idx2word = None
        self.is_training = False
        self.data_path = "data"
        
        self.root = Tk()
        self.root.title("MppleAI - NumPy LSTM Chat")
        self.root.geometry("700x800")
        self.root.configure(bg="#1e1e1e")
        
        self.setup_ui()
    
    def setup_ui(self):
        title = Label(self.root, text="MppleAI Chatbot (NumPy)", font=("Arial", 20, "bold"),
                     bg="#1e1e1e", fg="#00ff00")
        title.pack(pady=5)
        
        self.info_label = Label(self.root, text="Pure NumPy - No PyTorch Required!", font=("Arial", 10),
                               bg="#1e1e1e", fg="#00ff88")
        self.info_label.pack()
        
        train_frame = LabelFrame(self.root, text="Training", bg="#2d2d2d", fg="#ffffff", padx=10, pady=5)
        train_frame.pack(fill=X, padx=10, pady=5)
        
        btn_row = Frame(train_frame, bg="#2d2d2d")
        btn_row.pack()
        
        Button(btn_row, text="Load Data", command=self.load_data, bg="#555555", fg="white", width=12).pack(side=LEFT, padx=2)
        Button(btn_row, text="Start", command=self.start_training, bg="#008800", fg="white", width=12).pack(side=LEFT, padx=2)
        Button(btn_row, text="Stop", command=self.stop_training, bg="#880000", fg="white", width=12, state=DISABLED).pack(side=LEFT, padx=2)
        
        self.stats_label = Label(train_frame, text="No data loaded", bg="#2d2d2d", fg="#aaaaaa")
        self.stats_label.pack()
        
        self.progress = Label(train_frame, text="", bg="#2d2d2d", fg="#00ff00")
        self.progress.pack()
        
        self.loss_label = Label(train_frame, text="Best Loss: --", bg="#2d2d2d", fg="#00ff00")
        self.loss_label.pack()
        
        self.chat = scrolledtext.ScrolledText(self.root, height=22, width=75, 
                                             font=("Consolas", 10), bg="#2d2d2d", fg="#ffffff",
                                             insertbackground="#ffffff")
        self.chat.pack(padx=10, pady=5)
        
        input_frame = Frame(self.root, bg="#1e1e1e")
        input_frame.pack(fill=X, padx=10, pady=5)
        
        self.input_box = Entry(input_frame, font=("Arial", 14), bg="#3d3d3d", fg="#ffffff",
                             insertbackground="#ffffff")
        self.input_box.pack(side=LEFT, fill=X, expand=True)
        self.input_box.bind("<Return>", self.send_message)
        
        Button(input_frame, text="Send", command=self.send_message, bg="#0066aa", fg="white", 
               font=("Arial", 12), width=10).pack(side=RIGHT)
    
    def log(self, msg):
        self.chat.config(state=NORMAL)
        self.chat.insert(END, msg + "\n")
        self.chat.config(state=DISABLED)
        self.chat.see(END)
    
    def load_data(self):
        path = filedialog.askdirectory(title="Select data folder")
        if not path:
            return
        
        self.data_path = path
        self.log(f"Loading: {path}")
        
        lines_file = os.path.join(path, 'movie_lines.txt')
        convs_file = os.path.join(path, 'movie_conversations.txt')
        
        if not os.path.exists(lines_file):
            self.log("Error: movie_lines.txt not found!")
            return
        
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
        
        self.conversations = [(s, t) for s, t in conversations if s and t]
        self.log(f"Loaded {len(self.conversations)} pairs")
        
        all_text = [s for s, t in self.conversations] + [t for s, t in self.conversations]
        self.vocab, vocab_size = build_vocab(all_text)
        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.log(f"Vocab: {vocab_size}")
        
        max_len = 50
        self.src_seqs = np.array([encode(s, self.vocab, max_len) for s, _ in self.conversations])
        self.tgt_seqs = np.array([encode(t, self.vocab, max_len) for _, t in self.conversations])
        
        self.log(f"Ready! {len(self.src_seqs)} samples")
        self.stats_label.config(text=f"Data: {len(self.conversations)} pairs")
    
    def start_training(self):
        if self.is_training:
            return
        if not hasattr(self, 'conversations'):
            self.log("Load data first!")
            return
        
        self.is_training = True
        self.thread = threading.Thread(target=self.train_loop)
        self.thread.daemon = True
        self.thread.start()
        self.log("Training started...")
    
    def stop_training(self):
        self.is_training = False
        self.log("Stopped.")
    
    def train_loop(self):
        self.model = NumPyLSTM(len(self.vocab), embed_size=64, hidden_size=128)
        
        epoch = 0
        best_loss = float('inf')
        batch_size = 64
        lr = 0.002
        model_path = "chat_model.pkl"
        
        self.log(f"Model ready. Params: ~{self.model.W_embed.size + self.model.W_out.size + self.model.W_ih.size}")
        
        while self.is_training:
            epoch += 1
            total_loss = 0
            batches = 0
            start = time.time()
            
            indices = np.random.permutation(len(self.src_seqs))
            
            for i in range(0, len(self.src_seqs), batch_size):
                idx = indices[i:i + batch_size]
                src = self.src_seqs[idx]
                tgt = self.tgt_seqs[idx]
                
                logits, states = self.model.forward(src)
                loss, probs = self.model.compute_loss(logits, tgt)
                self.model.backward(src, tgt, probs, states)
                self.model.update(lr)
                
                total_loss += loss
                batches += 1
            
            avg_loss = total_loss / batches
            elapsed = time.time() - start
            
            self.stats_label.config(text=f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.loss_label.config(text=f"Best: {best_loss:.4f}")
                self.model.save(model_path, self.vocab, self.idx2word, epoch, avg_loss)
                self.log(f"[SAVE] {epoch} | {avg_loss:.4f}")
    
    def send_message(self, event=None):
        if self.model is None:
            self.log("Train first!")
            return
        
        text = self.input_box.get().strip()
        if not text:
            return
        
        self.input_box.delete(0, END)
        self.log(f"You: {text}")
        
        try:
            response = generate(self.model, text, self.vocab, self.idx2word)
            self.log(f"Bot: {response}")
        except Exception as e:
            self.log(f"Error: {e}")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("Starting MppleAI (NumPy version)...")
    app = MppleAI()
    app.run()
