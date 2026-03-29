# -*- coding: utf-8 -*-
"""
RNN 对话模型 - 使用 Cornell Movie Dialogs 数据集
"""

import os
import re
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

# ==================== 2. RNN 模型 ====================
class SimpleRNN:
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # 初始化参数
        scale = 0.1
        self.W_embed = xp.random.randn(vocab_size, embed_size) * scale
        self.W_ih = xp.random.randn(embed_size, hidden_size) * scale
        self.W_hh = xp.random.randn(hidden_size, hidden_size) * scale
        self.b_h = xp.zeros(hidden_size)
        self.W_ho = xp.random.randn(hidden_size, vocab_size) * scale
        self.b_o = xp.zeros(vocab_size)
        
        # 存储梯度
        self.grads = {}
        
    def forward(self, x, h=None):
        """前向传播"""
        batch, seq_len = x.shape
        if h is None:
            h = xp.zeros((batch, self.hidden_size))
        
        outputs = []
        hs = []
        embeds = []
        
        for t in range(seq_len):
            x_t = x[:, t].astype(int)
            embed = self.W_embed[x_t]
            embeds.append(embed)
            h = xp.tanh(embed @ self.W_ih + h @ self.W_hh + self.b_h)
            hs.append(h)
            out = h @ self.W_ho + self.b_o
            outputs.append(out)
        
        return xp.stack(outputs, axis=1), hs, embeds
    
    def compute_loss(self, logits, targets):
        """计算交叉熵损失"""
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1).astype(int)
        
        # Softmax
        max_val = xp.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = xp.exp(logits_flat - max_val)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)
        
        # 交叉熵
        correct_probs = probs[xp.arange(len(targets_flat)), targets_flat]
        loss = -xp.mean(xp.log(correct_probs + 1e-8))
        return float(loss), probs.reshape(batch, seq_len, vocab_size)
    
    def backward(self, x, y, probs, hs, embeds):
        """手动反向传播计算梯度"""
        batch, seq_len = x.shape
        vocab_size = self.vocab_size
        
        # 初始化梯度
        grad_W_ho = xp.zeros_like(self.W_ho)
        grad_b_o = xp.zeros_like(self.b_o)
        grad_W_ih = xp.zeros_like(self.W_ih)
        grad_W_hh = xp.zeros_like(self.W_hh)
        grad_b_h = xp.zeros_like(self.b_h)
        grad_W_embed = xp.zeros_like(self.W_embed)
        
        # 计算输出层梯度
        y_flat = y.reshape(-1).astype(int)
        
        # 输出层梯度
        dlogits = probs.copy()
        dlogits_flat = dlogits.reshape(-1, vocab_size)
        dlogits_flat[xp.arange(len(y_flat)), y_flat] -= 1
        dlogits = dlogits_flat.reshape(batch, seq_len, vocab_size) / (batch * seq_len)
        
        # 反向时间步
        dh_next = xp.zeros((batch, self.hidden_size))
        
        for t in reversed(range(seq_len)):
            # 输出层梯度
            grad_W_ho += hs[t].T @ dlogits[:, t, :]
            grad_b_o += xp.sum(dlogits[:, t, :], axis=0)
            
            # 隐藏层梯度
            dh = dlogits[:, t, :] @ self.W_ho.T + dh_next
            
            # tanh 梯度
            dh = dh * (1 - hs[t] ** 2)
            
            # RNN 参数梯度
            grad_W_ih += embeds[t].T @ dh
            if t > 0:
                grad_W_hh += hs[t-1].T @ dh
            grad_b_h += xp.sum(dh, axis=0)
            
            # 传递给前一时刻
            dh_next = dh @ self.W_hh.T
            
            # Embedding 梯度
            grad_embed_input = dh @ self.W_ih.T
            indices = x[:, t].astype(int)
            for i in range(batch):
                grad_W_embed[indices[i]] += grad_embed_input[i]
        
        # 存储梯度
        self.grads = {
            'W_embed': grad_W_embed,
            'W_ih': grad_W_ih,
            'W_hh': grad_W_hh,
            'b_h': grad_b_h,
            'W_ho': grad_W_ho,
            'b_o': grad_b_o,
        }
    
    def update(self, lr):
        """更新参数"""
        self.W_embed -= lr * self.grads['W_embed']
        self.W_ih -= lr * self.grads['W_ih']
        self.W_hh -= lr * self.grads['W_hh']
        self.b_h -= lr * self.grads['b_h']
        self.W_ho -= lr * self.grads['W_ho']
        self.b_o -= lr * self.grads['b_o']
    
    def save(self, filepath, vocab, idx2word, epoch, loss):
        """保存模型"""
        state = {
            'epoch': epoch,
            'loss': loss,
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'W_embed': self.W_embed,
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'W_ho': self.W_ho,
            'b_o': self.b_o,
            'vocab': vocab,
            'idx2word': idx2word,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"  ✓ Saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.vocab_size = state['vocab_size']
        self.embed_size = state['embed_size']
        self.hidden_size = state['hidden_size']
        self.W_embed = state['W_embed']
        self.W_ih = state['W_ih']
        self.W_hh = state['W_hh']
        self.b_h = state['b_h']
        self.W_ho = state['W_ho']
        self.b_o = state['b_o']
        
        return state['vocab'], state['idx2word'], state['epoch'], state['loss']


# ==================== 3. 数据处理 ====================
def clean_text(text):
    """清洗文本"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vocab(texts, min_freq=2, max_vocab=10000):
    """构建词汇表"""
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
    """将句子编码为 token 序列"""
    tokens = ['<SOS>'] + sentence.split() + ['<EOS>']
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    return np.array(indices, dtype=np.int32)


# ==================== 4. 数据加载器 ====================
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
            yield src_batch, tgt_batch

    def __len__(self):
        return int(np.ceil(len(self.src_seqs) / self.batch_size))


# ==================== 5. 加载 Cornell Movie Dialogs ====================
def load_cornell_movie_dialogs(data_dir):
    """加载 Cornell Movie Dialogs 数据集"""
    lines_file = os.path.join(data_dir, 'movie_lines.txt')
    convs_file = os.path.join(data_dir, 'movie_conversations.txt')
    
    if not os.path.exists(lines_file) or not os.path.exists(convs_file):
        print(f"Dataset not found in {data_dir}")
        return None
    
    print("Reading movie lines...")
    lines = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
    
    print("Reading conversations...")
    conversations = []
    with open(convs_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 4:
                utterance_ids = eval(parts[3])
                for i in range(len(utterance_ids) - 1):
                    src = lines.get(utterance_ids[i], '')
                    tgt = lines.get(utterance_ids[i + 1], '')
                    if src and tgt and len(src.split()) > 1 and len(tgt.split()) > 1:
                        conversations.append((src, tgt))
    
    print(f"Loaded {len(conversations)} conversation pairs")
    return conversations


# ==================== 6. 训练函数 ====================
def train_model(model, train_loader, epochs, lr, vocab, idx2word, save_path='rnn_model.pkl', start_epoch=0):
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for src, tgt in train_loader:
            src_gpu = xp.asarray(src)
            tgt_gpu = xp.asarray(tgt)
            
            # 前向传播
            logits, hs, embeds = model.forward(src_gpu)
            loss, probs = model.compute_loss(logits, tgt_gpu)
            
            # 反向传播
            model.backward(src_gpu, tgt_gpu, probs, hs, embeds)
            model.update(lr)
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(save_path, vocab, idx2word, epoch, avg_loss)
    
    return model


# ==================== 7. 生成回复 ====================
def generate_response(model, input_sentence, vocab, idx2word, max_len=15, temperature=0.5):
    input_tokens = encode_sentence(input_sentence, vocab, max_len)
    src = xp.asarray(input_tokens.reshape(1, -1))
    
    generated = []
    h = None
    last_token = None
    repeat_count = 0
    pad_token = vocab['<PAD>']
    eos_token = vocab['<EOS>']
    
    for _ in range(max_len):
        logits, hs, _ = model.forward(src, h)
        last_logits = logits[0, -1, :]
        
        # 应用温度
        last_logits = last_logits / temperature
        
        # 屏蔽 <PAD>
        last_logits[pad_token] = -float('inf')
        
        # Softmax
        max_val = xp.max(last_logits)
        exp_logits = xp.exp(last_logits - max_val)
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
        
        if next_token == eos_token:
            break
        generated.append(next_token)
        
        # 更新输入
        new_input = np.append(src[0][1:], next_token)
        src = xp.asarray(new_input.reshape(1, -1))
        h = hs[-1] if hs else None
    
    response = ' '.join([idx2word.get(t, '<UNK>') for t in generated])
    return response if response else "hello"


# ==================== 8. 主函数 ====================
def main():
    print("=" * 50)
    print("RNN Chatbot - Cornell Movie Dialogs")
    print("=" * 50)
    
    model_file = 'rnn_model.pkl'
    data_dir = 'data'
    
    # 加载数据集
    print("\nLoading Cornell Movie Dialogs...")
    conversations = load_cornell_movie_dialogs(data_dir)
    
    if conversations is None or len(conversations) == 0:
        print("Failed to load dataset. Using sample dialogues...")
        # 使用示例对话作为后备
        dialogues = [
            "hello", "hi there",
            "hi", "hello how are you",
            "how are you", "i am fine thank you",
            "what is your name", "my name is chatbot",
            "what can you do", "i can answer questions",
            "goodbye", "see you later",
        ]
        conversations = []
        for i in range(0, len(dialogues) - 1, 2):
            if i + 1 < len(dialogues):
                conversations.append((dialogues[i], dialogues[i + 1]))
    
    # 限制数据量（先用 10000 对，可以调整）
    max_pairs = 10000
    if len(conversations) > max_pairs:
        conversations = conversations[:max_pairs]
        print(f"Using first {max_pairs} conversation pairs")
    
    # 清洗数据
    print("Cleaning data...")
    cleaned = []
    for src, tgt in conversations:
        src_clean = clean_text(src)
        tgt_clean = clean_text(tgt)
        if src_clean and tgt_clean and len(src_clean.split()) > 1 and len(tgt_clean.split()) > 1:
            cleaned.append((src_clean, tgt_clean))
    
    print(f"Cleaned {len(cleaned)} conversation pairs")
    
    # 构建词表
    print("Building vocabulary...")
    all_texts = []
    for src, tgt in cleaned:
        all_texts.append(src)
        all_texts.append(tgt)
    
    vocab, vocab_size = build_vocab(all_texts, min_freq=2, max_vocab=5000)
    idx2word = {v: k for k, v in vocab.items()}
    print(f"Vocabulary size: {vocab_size}")
    
    # 编码数据
    print("Encoding data...")
    max_len = 15
    src_seqs = []
    tgt_seqs = []
    for src, tgt in cleaned:
        src_enc = encode_sentence(src, vocab, max_len)
        tgt_enc = encode_sentence(tgt, vocab, max_len)
        src_seqs.append(src_enc)
        tgt_seqs.append(tgt_enc)
    
    # 转换为数组
    src_seqs_np = np.array(src_seqs)
    tgt_seqs_np = np.array(tgt_seqs)
    
    batch_size = 32
    train_loader = DataLoader(src_seqs_np, tgt_seqs_np, batch_size=batch_size)
    
    # 创建模型
    print("\nCreating model...")
    model = SimpleRNN(vocab_size, embed_size=64, hidden_size=128)
    total_params = (vocab_size * 64) + (64 * 128) + (128 * 128) + 128 + (128 * vocab_size) + vocab_size
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
                vocab, idx2word, last_epoch, last_loss = model.load(model_file)
                print(f"Continuing from epoch {last_epoch + 1} (Loss: {last_loss:.4f})")
                # 这里应该调用 train_model，但可能被注释或跳过了
                model = train_model(model, train_loader, epochs=100, lr=0.005, 
                                vocab=vocab, idx2word=idx2word, 
                                save_path=model_file, start_epoch=last_epoch + 1)
            except Exception as e:
                print(f"Error: {e}, training new model")
                model = train_model(model, train_loader, epochs=100, lr=0.005, 
                                vocab=vocab, idx2word=idx2word, save_path=model_file)
        else:
            print("Loading model for chat...")
            try:
                vocab, idx2word, _, _ = model.load(model_file)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training new model...")
                model = train_model(model, train_loader, epochs=100, lr=0.005, 
                                   vocab=vocab, idx2word=idx2word, save_path=model_file)
    else:
        print("\nNo existing model found. Starting training...")
        model = train_model(model, train_loader, epochs=100, lr=0.005, 
                           vocab=vocab, idx2word=idx2word, save_path=model_file)
    
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
        
        response = generate_response(model, user_input, vocab, idx2word, max_len=15, temperature=0.5)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()