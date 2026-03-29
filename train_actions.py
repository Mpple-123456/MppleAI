import os
import sys
import time
import argparse
import numpy as np
import pickle
import re
from collections import Counter

# ------------------------------
# 文本处理函数
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vocab(texts, min_freq=2, max_vocab=5000):
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

# ------------------------------
# RNN 模型类（手动反向传播）
# ------------------------------
class SimpleRNN:
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        scale = 0.1
        self.W_embed = np.random.randn(vocab_size, embed_size) * scale
        self.W_ih = np.random.randn(embed_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        self.W_ho = np.random.randn(hidden_size, vocab_size) * scale
        self.b_o = np.zeros(vocab_size)
        self.grads = {}

    def forward(self, x, h=None):
        batch, seq_len = x.shape
        if h is None:
            h = np.zeros((batch, self.hidden_size))
        outputs = []
        hs = []
        embeds = []
        for t in range(seq_len):
            x_t = x[:, t].astype(int)
            embed = self.W_embed[x_t]
            embeds.append(embed)
            h = np.tanh(embed @ self.W_ih + h @ self.W_hh + self.b_h)
            hs.append(h)
            out = h @ self.W_ho + self.b_o
            outputs.append(out)
        return np.stack(outputs, axis=1), hs, embeds

    def compute_loss(self, logits, targets):
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1).astype(int)
        max_val = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_val)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
        loss = -np.mean(np.log(correct_probs + 1e-8))
        return float(loss), probs.reshape(batch, seq_len, vocab_size)

    def backward(self, x, y, probs, hs, embeds):
        batch, seq_len = x.shape
        vocab_size = self.vocab_size
        grad_W_ho = np.zeros_like(self.W_ho)
        grad_b_o = np.zeros_like(self.b_o)
        grad_W_ih = np.zeros_like(self.W_ih)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_b_h = np.zeros_like(self.b_h)
        grad_W_embed = np.zeros_like(self.W_embed)

        y_flat = y.reshape(-1).astype(int)
        dlogits = probs.copy()
        dlogits_flat = dlogits.reshape(-1, vocab_size)
        dlogits_flat[np.arange(len(y_flat)), y_flat] -= 1
        dlogits = dlogits_flat.reshape(batch, seq_len, vocab_size) / (batch * seq_len)

        dh_next = np.zeros((batch, self.hidden_size))
        for t in reversed(range(seq_len)):
            grad_W_ho += hs[t].T @ dlogits[:, t, :]
            grad_b_o += np.sum(dlogits[:, t, :], axis=0)
            dh = dlogits[:, t, :] @ self.W_ho.T + dh_next
            dh = dh * (1 - hs[t] ** 2)
            grad_W_ih += embeds[t].T @ dh
            if t > 0:
                grad_W_hh += hs[t-1].T @ dh
            grad_b_h += np.sum(dh, axis=0)
            dh_next = dh @ self.W_hh.T
            grad_embed_input = dh @ self.W_ih.T
            indices = x[:, t].astype(int)
            for i in range(batch):
                grad_W_embed[indices[i]] += grad_embed_input[i]

        self.grads = {
            'W_embed': grad_W_embed,
            'W_ih': grad_W_ih,
            'W_hh': grad_W_hh,
            'b_h': grad_b_h,
            'W_ho': grad_W_ho,
            'b_o': grad_b_o,
        }

    def update(self, lr):
        self.W_embed -= lr * self.grads['W_embed']
        self.W_ih -= lr * self.grads['W_ih']
        self.W_hh -= lr * self.grads['W_hh']
        self.b_h -= lr * self.grads['b_h']
        self.W_ho -= lr * self.grads['W_ho']
        self.b_o -= lr * self.grads['b_o']

    def save(self, filepath):
        state = {
            'W_embed': self.W_embed,
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'W_ho': self.W_ho,
            'b_o': self.b_o,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.W_embed = state['W_embed']
        self.W_ih = state['W_ih']
        self.W_hh = state['W_hh']
        self.b_h = state['b_h']
        self.W_ho = state['W_ho']
        self.b_o = state['b_o']

# ------------------------------
# 数据加载器
# ------------------------------
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

# ------------------------------
# 加载 Cornell Movie Dialogs
# ------------------------------
def load_conversations(data_dir):
    lines_file = os.path.join(data_dir, 'movie_lines.txt')
    convs_file = os.path.join(data_dir, 'movie_conversations.txt')
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
                utterance_ids = eval(parts[3])
                for i in range(len(utterance_ids) - 1):
                    src = lines.get(utterance_ids[i], '')
                    tgt = lines.get(utterance_ids[i + 1], '')
                    if src and tgt:
                        conversations.append((src, tgt))
    return conversations

# ------------------------------
# 主训练函数（含早停、学习率衰减）
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--max_pairs', type=int, default=5000, help='使用的对话对数量')
    parser.add_argument('--lr', type=float, default=0.005, help='初始学习率')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--embed_size', type=int, default=32, help='嵌入维度')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值（连续多少个epoch损失不下降则停止）')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='学习率衰减因子（每10个epoch）')
    args = parser.parse_args()

    print("=" * 50)
    print("RNN Chatbot Training with Early Stopping & LR Decay")
    print("=" * 50)
    print(f"参数: epochs={args.epochs}, max_pairs={args.max_pairs}, lr={args.lr}, hidden_size={args.hidden_size}, patience={args.patience}")

    # 加载数据
    data_dir = 'data'
    if not os.path.exists(os.path.join(data_dir, 'movie_lines.txt')):
        print("错误：未找到数据文件，请先运行下载步骤")
        sys.exit(1)

    conversations = load_conversations(data_dir)
    conversations = conversations[:args.max_pairs]
    print(f"加载了 {len(conversations)} 对对话")

    # 清洗数据
    cleaned = []
    for src, tgt in conversations:
        src_clean = clean_text(src)
        tgt_clean = clean_text(tgt)
        if src_clean and tgt_clean:
            cleaned.append((src_clean, tgt_clean))
    print(f"清洗后剩余 {len(cleaned)} 对")

    # 构建词汇表
    all_texts = []
    for src, tgt in cleaned:
        all_texts.append(src)
        all_texts.append(tgt)
    vocab, vocab_size = build_vocab(all_texts, min_freq=2, max_vocab=5000)
    print(f"词汇表大小: {vocab_size}")

    # 编码数据
    max_len = 15
    src_seqs = []
    tgt_seqs = []
    for src, tgt in cleaned:
        src_enc = encode_sentence(src, vocab, max_len)
        tgt_enc = encode_sentence(tgt, vocab, max_len)
        src_seqs.append(src_enc)
        tgt_seqs.append(tgt_enc)
    src_seqs_np = np.array(src_seqs)
    tgt_seqs_np = np.array(tgt_seqs)

    # 划分训练集和验证集（90% 训练，10% 验证）
    split = int(0.9 * len(src_seqs_np))
    train_src, val_src = src_seqs_np[:split], src_seqs_np[split:]
    train_tgt, val_tgt = tgt_seqs_np[:split], tgt_seqs_np[split:]

    batch_size = 32
    train_loader = DataLoader(train_src, train_tgt, batch_size=batch_size)
    val_loader = DataLoader(val_src, val_tgt, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = SimpleRNN(vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size)
    print("模型创建完成")

    # 训练准备
    best_loss = float('inf')
    wait = 0
    lr = args.lr
    best_epoch = 0

    print(f"开始训练，最多 {args.epochs} 个 epoch，早停耐心值 {args.patience}")

    for epoch in range(args.epochs):
        # 学习率衰减：每 10 个 epoch 乘以 decay 因子
        if epoch > 0 and epoch % 10 == 0:
            lr *= args.lr_decay
            print(f"学习率衰减至 {lr:.6f}")

        # 训练一个 epoch
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        for src, tgt in train_loader:
            logits, hs, embeds = model.forward(src)
            loss, probs = model.compute_loss(logits, tgt)
            model.backward(src, tgt, probs, hs, embeds)
            model.update(lr)
            total_loss += loss
            num_batches += 1
        train_loss = total_loss / num_batches

        # 验证一个 epoch
        val_loss = 0
        num_val = 0
        for src, tgt in val_loader:
            logits, hs, embeds = model.forward(src)
            loss, _ = model.compute_loss(logits, tgt)
            val_loss += loss
            num_val += 1
        val_loss /= num_val

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s | LR: {lr:.5f}")

        # 早停判断（使用验证损失）
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            model.save('best_model.pkl')
            print(f"  -> 新最佳模型 (损失 {best_loss:.4f}) 已保存")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"早停触发！验证损失连续 {args.patience} 个 epoch 未下降，停止训练。")
                break

        # 定期保存中间模型（每 10 个 epoch）
        if (epoch + 1) % 10 == 0:
            model.save(f'model_epoch_{epoch+1}.pkl')

    # 训练结束，加载最佳模型
    model.load('best_model.pkl')
    model.save('rnn_model.pkl')
    print(f"训练完成。最佳验证损失 {best_loss:.4f} 出现在 epoch {best_epoch+1}")
    print("最终模型已保存为 rnn_model.pkl")

if __name__ == "__main__":
    main()
