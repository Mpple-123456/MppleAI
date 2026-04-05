# -*- coding: utf-8 -*-
"""
聊天工具 - 加载训练好的模型
"""

import os
import numpy as np
import torch
import torch.nn as nn

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
        self.embed_size = embed_size
    
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


def generate(model, text, vocab, idx2word, max_len=30, temperature=0.3):
    model.eval()
    src = torch.LongTensor(encode(text, vocab, max_len)).unsqueeze(0).to(device)
    hidden = None
    result = []
    last = None
    seen = set()
    eos = vocab['<EOS>']
    pad = vocab['<PAD>']
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(src, hidden)
            logits = logits[0, -1] / temperature
            logits[pad] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            
            if token == eos:
                break
            if last and (last, token) in seen:
                break
            seen.add((last, token))
            last = token
            word = idx2word.get(token, '')
            if word and word not in ['<PAD>', '<SOS>', '<UNK>']:
                result.append(word)
            
            src = torch.cat([src[0, 1:].cpu(), torch.tensor([token])]).unsqueeze(0).to(device)
    
    return ' '.join(result) if result else "i am not sure how to respond"


def main():
    model_path = 'chat_model.pkl'
    
    print("\n" + "=" * 50)
    print("Chatbot - AI Chat")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please run 'python train_infinite.py' first to train the model.")
        return
    
    print("\n[LOAD] Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    vocab_size = checkpoint['model_state']['embedding.weight'].shape[0]
    embed_size = checkpoint['model_state']['embedding.weight'].shape[1]
    hidden_size = checkpoint['model_state']['lstm.weight_hh_l0'].shape[0] // 4
    num_layers = 2
    
    model = ChatLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.load_state_dict(checkpoint['model_state'])
    vocab = checkpoint['vocab']
    idx2word = checkpoint['idx2word']
    
    print(f"[OK] Model loaded!")
    print(f"     Epoch: {checkpoint['epoch']}")
    print(f"     Loss: {checkpoint['loss']:.4f}")
    print(f"     Vocab: {len(vocab)} words")
    print(f"     Device: {device}")
    
    print("\n" + "=" * 50)
    print("Chat ready! Type 'exit' to quit.")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user = input("You: ")
            if user.lower() == 'exit':
                break
            if not user.strip():
                continue
            
            response = generate(model, user, vocab, idx2word)
            print(f"Bot: {response}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nSession ended.")


if __name__ == "__main__":
    main()
