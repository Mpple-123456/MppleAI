# train_more.py - 继续训练脚本
import pickle
import time
import numpy as np
import cupy as xp

# 加载已有模型
print("Loading model...")
with open('rnn_model.pkl', 'rb') as f:
    state = pickle.load(f)

print(f"Model info: Epoch {state['epoch']+1}, Loss: {state['loss']:.4f}")
print(f"Vocab size: {state['vocab_size']}")
print(f"Hidden size: {state['hidden_size']}")

# 重新创建模型并加载参数
from simple_rnn_chat import SimpleRNN, DataLoader, train_model, clean_text, encode_sentence, build_vocab

model = SimpleRNN(state['vocab_size'], state['embed_size'], state['hidden_size'])
for p, saved_p in zip(model.parameters(), state['params']):
    p.data = saved_p

vocab = state['vocab']
idx2word = state['idx2word']

# 准备数据（使用相同的对话数据）
dialogues = [
    "hello", "hi there",
    "hi", "hello how are you",
    "how are you", "i am fine thank you",
    "what is your name", "my name is chatbot",
    "what can you do", "i can answer questions",
    "tell me a joke", "why did the chicken cross the road",
    "goodbye", "see you later",
    "thank you", "you are welcome",
    "how is the weather", "it is sunny today",
    "do you like music", "yes i love music",
    "what is your hobby", "i enjoy reading",
    "where are you from", "i am from the internet",
]

conversations = []
for i in range(0, len(dialogues) - 1, 2):
    if i + 1 < len(dialogues):
        conversations.append((dialogues[i], dialogues[i + 1]))

cleaned = [(clean_text(src), clean_text(tgt)) for src, tgt in conversations]

all_texts = []
for src, tgt in cleaned:
    all_texts.append(src)
    all_texts.append(tgt)

max_len = 12
src_seqs = []
tgt_seqs = []
for src, tgt in cleaned:
    src_enc = encode_sentence(src, vocab, max_len)
    tgt_enc = encode_sentence(tgt, vocab, max_len)
    src_seqs.append(src_enc)
    tgt_seqs.append(tgt_enc)

src_seqs_gpu = xp.asarray(src_seqs)
tgt_seqs_gpu = xp.asarray(tgt_seqs)

batch_size = 8
train_loader = DataLoader(src_seqs_gpu, tgt_seqs_gpu, batch_size=batch_size)

# 继续训练
print("\nContinuing training for 200 more epochs...")
train_model(model, train_loader, epochs=200, lr=0.001, save_path='rnn_model.pkl', start_epoch=state['epoch'] + 1)

print("\nTraining complete!")