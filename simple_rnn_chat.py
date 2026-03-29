# -*- coding: utf-8 -*-
"""
RNN 对话模型 - 完整修复版
使用简化但正确的自动微分
"""

import os
import re
import math
import time
import pickle
import numpy as np
from collections import Counter

# ==================== 1. 后端配置 ====================
try:
    import cupy as xp
    print("=" * 50)
    print("Using GPU (CuPy)")
    try:
        device = xp.cuda.Device(0)
        free, total = device.mem_info
        print(f"GPU Memory: {(total - free) / 1024**3:.1f}GB / {total / 1024**3:.1f}GB")
    except:
        pass
    print("=" * 50)
except ImportError:
    import numpy as xp
    print("Using CPU (NumPy)")

# ==================== 2. Tensor 类（简化正确版）====================
class Tensor:
    def __init__(self, data, requires_grad=False, depends_on=None):
        self.data = xp.asarray(data, dtype=xp.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.depends_on = depends_on if depends_on is not None else []

    def backward(self, grad=None):
        if grad is None:
            grad = xp.ones_like(self.data)
        self.grad = grad
        for dep in self.depends_on:
            if dep.requires_grad:
                dep.backward(grad)

    def zero_grad(self):
        self.grad = None
        for dep in self.depends_on:
            dep.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        """重塑张量形状"""
        return Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, depends_on=[self])

    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad, depends_on=[self])

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, 
                      depends_on=[self, other])

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                      depends_on=[self, other])

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


# ==================== 3. 线性层 ====================
class Linear:
    def __init__(self, in_features, out_features):
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(xp.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.bias = Tensor(xp.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


# ==================== 4. RNN 单元 ====================
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = 0.01
        self.W_ih = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True)
        self.W_hh = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True)
        self.bias = Tensor(xp.zeros(hidden_size), requires_grad=True)

    def forward(self, x, h):
        gate = x @ self.W_ih + h @ self.W_hh + self.bias
        new_h = Tensor(xp.tanh(gate.data), requires_grad=True, depends_on=[gate])
        return new_h

    def parameters(self):
        return [self.W_ih, self.W_hh, self.bias]


# ==================== 5. Embedding 层 ====================
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor(xp.random.uniform(-0.1, 0.1, (num_embeddings, embedding_dim)), 
                             requires_grad=True)

    def forward(self, x):
        indices = x.data.astype(int)
        return Tensor(self.weight.data[indices], requires_grad=self.weight.requires_grad, 
                      depends_on=[self.weight])

    def parameters(self):
        return [self.weight]


# ==================== 6. RNN 模型 ====================
class SimpleRNN:
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embed_size)
        self.rnn = RNNCell(embed_size, hidden_size)
        self.fc = Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        batch, seq_len = x.shape
        if h is None:
            h = Tensor(xp.zeros((batch, self.hidden_size)), requires_grad=False)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            embed = self.embedding.forward(x_t)
            h = self.rnn.forward(embed, h)
            out = self.fc.forward(h)
            outputs.append(out)

        stacked = xp.stack([o.data for o in outputs], axis=1)
        return Tensor(stacked, requires_grad=x.requires_grad), h

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.rnn.parameters())
        params.extend(self.fc.parameters())
        return params


# ==================== 7. 损失函数 ====================
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    # 直接使用 .data 获取数组，然后 reshape
    logits_flat = logits.data.reshape(batch * seq_len, vocab_size)
    targets_flat = targets.data.reshape(batch * seq_len).astype(int)
    
    # Softmax（使用数组操作）
    max_val = xp.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = xp.exp(logits_flat - max_val)
    probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)
    
    # 交叉熵
    correct_probs = probs[xp.arange(batch * seq_len), targets_flat]
    loss_val = -xp.mean(xp.log(correct_probs + 1e-8))
    
    loss = Tensor(loss_val, requires_grad=logits.requires_grad, depends_on=[logits])
    return loss


# ==================== 8. SGD 优化器 ====================
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ==================== 9. 数据处理 ====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vocab(texts, min_freq=1, max_vocab=500):
    counter = Counter()
    for text in texts:
        for word in text.split():
            counter[word] += 1
    
    most_common = counter.most_common(max_vocab - 4)
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for word, freq in most_common:
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab, len(vocab)

def encode_sentence(sentence, vocab, max_len):
    tokens = ['<SOS>'] + sentence.split() + ['<EOS>']
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    return np.array(indices, dtype=np.int32)


# ==================== 10. 数据加载器 ====================
class DataLoader:
    def __init__(self, src_seqs, tgt_seqs, batch_size, shuffle=True):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(src_seqs))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.src_seqs), self.batch_size):
            idx = self.indices[start:start + self.batch_size]
            src_batch = np.stack([self.src_seqs[i] for i in idx])
            tgt_batch = np.stack([self.tgt_seqs[i] for i in idx])
            yield Tensor(src_batch), Tensor(tgt_batch)

    def __len__(self):
        return int(np.ceil(len(self.src_seqs) / self.batch_size))


# ==================== 11. 模型保存加载 ====================
def save_model(model, vocab, idx2word, filepath, epoch, loss):
    model_state = {
        'epoch': epoch,
        'loss': loss,
        'vocab_size': model.vocab_size,
        'embed_size': model.embed_size,
        'hidden_size': model.hidden_size,
        'params': [p.data for p in model.parameters()],
        'vocab': vocab,
        'idx2word': idx2word,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"  ✓ Saved to {filepath}")

def load_model(filepath):
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    model = SimpleRNN(state['vocab_size'], state['embed_size'], state['hidden_size'])
    for p, saved_p in zip(model.parameters(), state['params']):
        p.data = saved_p
    
    vocab = state['vocab']
    idx2word = state['idx2word']
    
    print(f"  ✓ Loaded from {filepath} (Epoch {state['epoch']+1}, Loss: {state['loss']:.4f})")
    return model, vocab, idx2word, state['epoch'], state['loss']


# ==================== 12. 训练函数 ====================
def train_model(model, train_loader, epochs, lr, vocab, idx2word, save_path='rnn_model.pkl', start_epoch=0):
    optimizer = SGD(model.parameters(), lr=lr)
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for src, tgt in train_loader:
            # 前向
            logits, _ = model.forward(src)
            loss = cross_entropy_loss(logits, tgt)
            
            # 反向
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data.item()  # 这里用 .data.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, vocab, idx2word, save_path, epoch, avg_loss)
    
    return model


# ==================== 13. 生成回复 ====================
def generate_response(model, input_sentence, vocab, idx2word, max_len=15, temperature=0.5):
    if vocab is None:
        return "Error: Vocabulary not loaded"
    
    input_tokens = encode_sentence(input_sentence, vocab, max_len)
    src = Tensor(input_tokens.reshape(1, -1))
    
    generated = []
    h = None
    last_token = None
    repeat_count = 0
    
    for _ in range(max_len):
        logits, h = model.forward(src, h)
        last_logits = logits[:, -1, :]
        
        # 应用温度
        last_logits_data = last_logits.data[0] / temperature
        
        # Softmax
        max_val = xp.max(last_logits_data)
        exp_logits = xp.exp(last_logits_data - max_val)
        probs = exp_logits / xp.sum(exp_logits)
        
        # 贪心解码
        next_token = int(xp.argmax(probs))
        
        # 重复检测
        if next_token == last_token:
            repeat_count += 1
            if repeat_count >= 2:
                break
        else:
            repeat_count = 0
        last_token = next_token
        
        if next_token == vocab['<EOS>']:
            break
        generated.append(next_token)
        
        # 更新输入
        new_input = np.append(src.data[0][1:], next_token)
        src = Tensor(new_input.reshape(1, -1))
    
    response = ' '.join([idx2word.get(t, '<UNK>') for t in generated])
    return response if response else "i don't know"


# ==================== 14. 主函数 ====================
def main():
    print("=" * 50)
    print("RNN Chatbot - Fixed Version")
    print("=" * 50)
    
    model_file = 'rnn_model.pkl'
    
    # 对话数据
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
    ]
    
    # 构建对话对
    conversations = []
    for i in range(0, len(dialogues) - 1, 2):
        if i + 1 < len(dialogues):
            conversations.append((dialogues[i], dialogues[i + 1]))
    
    print(f"\nLoaded {len(conversations)} conversation pairs")
    
    # 清洗数据
    cleaned = [(clean_text(src), clean_text(tgt)) for src, tgt in conversations]
    
    # 构建词表
    all_texts = []
    for src, tgt in cleaned:
        all_texts.append(src)
        all_texts.append(tgt)
    
    vocab, vocab_size = build_vocab(all_texts, min_freq=1, max_vocab=100)
    idx2word = {v: k for k, v in vocab.items()}
    print(f"Vocabulary size: {vocab_size}")
    
    # 编码数据
    max_len = 8
    src_seqs = []
    tgt_seqs = []
    for src, tgt in cleaned:
        src_enc = encode_sentence(src, vocab, max_len)
        tgt_enc = encode_sentence(tgt, vocab, max_len)
        src_seqs.append(src_enc)
        tgt_seqs.append(tgt_enc)
    
    src_seqs_gpu = xp.asarray(src_seqs)
    tgt_seqs_gpu = xp.asarray(tgt_seqs)
    
    batch_size = 4
    train_loader = DataLoader(src_seqs_gpu, tgt_seqs_gpu, batch_size=batch_size)
    
    # 创建模型
    print("\nCreating model...")
    model = SimpleRNN(vocab_size, embed_size=32, hidden_size=64)
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {total_params / 1e3:.0f}K")
    
    # 检查是否有保存的模型
    if os.path.exists(model_file):
        print("\nFound existing model.")
        print("Options:")
        print("  1. Continue training")
        print("  2. Chat only")
        choice = input("Your choice (1/2): ").strip()
        
        if choice == '1':
            print("Loading model for continued training...")
            try:
                model, vocab, idx2word, last_epoch, last_loss = load_model(model_file)
                print(f"Continuing from epoch {last_epoch + 1}")
                model = train_model(model, train_loader, epochs=100, lr=0.01, 
                                   vocab=vocab, idx2word=idx2word,
                                   save_path=model_file, start_epoch=last_epoch + 1)
            except Exception as e:
                print(f"Error: {e}, training new model")
                model = train_model(model, train_loader, epochs=100, lr=0.01, 
                                   vocab=vocab, idx2word=idx2word, save_path=model_file)
        else:
            print("Loading model for chat...")
            try:
                model, vocab, idx2word, _, _ = load_model(model_file)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training new model...")
                model = train_model(model, train_loader, epochs=100, lr=0.01, 
                                   vocab=vocab, idx2word=idx2word, save_path=model_file)
    else:
        print("\nNo existing model found. Starting training...")
        model = train_model(model, train_loader, epochs=100, lr=0.01, 
                           vocab=vocab, idx2word=idx2word, save_path=model_file)
    
    # 保存最终模型
    save_model(model, vocab, idx2word, model_file, 99, 0)
    
    # 聊天
    print("\n" + "=" * 50)
    print("Chatbot ready! Type 'exit' to quit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
        
        response = generate_response(model, user_input, vocab, idx2word, max_len=12, temperature=0.5)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()