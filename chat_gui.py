# -*- coding: utf-8 -*-
"""
聊天 GUI - Tkinter
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tkinter import *
from tkinter import scrolledtext

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


def encode(sentence, vocab, max_len=50):
    chars = list(sentence)[:max_len-2]
    indices = [vocab.get(c, vocab['<UNK>']) for c in chars]
    indices = [vocab['<SOS>']] + indices + [vocab['<EOS>']]
    return np.array(indices + [vocab['<PAD>']] * (max_len - len(indices)), dtype=np.int64)


def generate(model, text, vocab, idx2word, max_len=50, temperature=0.15):
    model.eval()
    src = torch.LongTensor(encode(text, vocab, max_len)).unsqueeze(0).to(device)
    hidden = None
    result = []
    eos = vocab['<EOS>']
    pad = vocab['<PAD>']
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(src, hidden)
            logits = logits[0, -1] / temperature
            logits[pad] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            token = torch.argmax(probs).item()
            
            if token == eos:
                break
            word = idx2word.get(token, '')
            if word and word not in ['<PAD>', '<SOS>', '<UNK>']:
                result.append(word)
            
            src = torch.cat([src[0, 1:].cpu(), torch.tensor([token])]).unsqueeze(0).to(device)
    
    return ''.join(result) if result else "i am not sure how to respond"


class ChatGUI:
    def __init__(self):
        self.window = Tk()
        self.window.title("MppleAI - Chat")
        self.window.geometry("500x600")
        
        self.model = None
        self.vocab = None
        self.idx2word = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        self.info_label = Label(self.window, text="Device: " + ("GPU" if torch.cuda.is_available() else "CPU"))
        self.info_label.pack(pady=5)
        
        self.status_label = Label(self.window, text="Loading model...", fg="orange")
        self.status_label.pack(pady=5)
        
        self.chat_area = scrolledtext.ScrolledText(self.window, height=25, width=60, state=DISABLED)
        self.chat_area.pack(padx=20, pady=10)
        
        input_frame = Frame(self.window)
        input_frame.pack(fill=X, padx=20, pady=10)
        
        self.input_box = Entry(input_frame, font=("Arial", 12))
        self.input_box.pack(side=LEFT, fill=X, expand=True)
        self.input_box.bind("<Return>", self.send_message)
        
        self.send_btn = Button(input_frame, text="Send", command=self.send_message, bg="blue", fg="white", width=10)
        self.send_btn.pack(side=RIGHT, padx=5)
        
        btn_frame = Frame(self.window)
        btn_frame.pack(pady=5)
        
        self.clear_btn = Button(btn_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_btn.pack(side=LEFT, padx=5)
        
        self.reload_btn = Button(btn_frame, text="Reload Model", command=self.load_model)
        self.reload_btn.pack(side=LEFT, padx=5)
    
    def load_model(self):
        model_path = 'chat_model.pkl'
        
        if not os.path.exists(model_path):
            self.status_label.config(text="Model not found! Run train_gui.py first", fg="red")
            return
        
        try:
            self.status_label.config(text="Loading model...", fg="orange")
            self.window.update()
            
            checkpoint = torch.load(model_path, map_location=device)
            
            vocab_size = checkpoint['model_state']['embedding.weight'].shape[0]
            embed_size = checkpoint['model_state']['embedding.weight'].shape[1]
            hidden_size = checkpoint['model_state']['lstm.weight_hh_l0'].shape[0] // 4
            num_layers = 2
            
            self.model = ChatLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.vocab = checkpoint['vocab']
            self.idx2word = checkpoint['idx2word']
            
            self.status_label.config(
                text=f"Model loaded! Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}",
                fg="green"
            )
            
            self.add_message("System", "Model loaded successfully! You can start chatting now.")
        
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {e}", fg="red")
    
    def add_message(self, sender, message):
        self.chat_area.config(state=NORMAL)
        self.chat_area.insert(END, f"\n{sender}: {message}\n")
        self.chat_area.config(state=DISABLED)
        self.chat_area.see(END)
    
    def send_message(self, event=None):
        if self.model is None:
            self.add_message("System", "Model not loaded!")
            return
        
        text = self.input_box.get().strip()
        if not text:
            return
        
        self.input_box.delete(0, END)
        self.add_message("You", text)
        
        try:
            response = generate(self.model, text, self.vocab, self.idx2word)
            self.add_message("Bot", response)
        except Exception as e:
            self.add_message("Bot", f"Error: {e}")
    
    def clear_chat(self):
        self.chat_area.config(state=NORMAL)
        self.chat_area.delete(1.0, END)
        self.chat_area.config(state=DISABLED)
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChatGUI()
    gui.run()
