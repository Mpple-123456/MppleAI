"""
GitHub Actions 自动训练脚本
"""

"""
GitHub Actions 自动训练脚本 - 自动下载数据
"""

import os
import sys
import urllib.request
import zipfile

def download_dataset():
    """自动下载 Cornell Movie Dialogs 数据集"""
    data_dir = 'data'
    
    if os.path.exists(os.path.join(data_dir, 'movie_lines.txt')):
        print("数据集已存在，跳过下载")
        return True
    
    print("正在下载数据集...")
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = "cornell_movie_dialogs_corpus.zip"
    
    try:
        # 下载文件
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成，正在解压...")
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # 移动文件
        os.makedirs(data_dir, exist_ok=True)
        os.system(f"mv cornell_movie_dialogs_corpus/movie_lines.txt {data_dir}/")
        os.system(f"mv cornell_movie_dialogs_corpus/movie_conversations.txt {data_dir}/")
        
        # 清理
        os.remove(zip_path)
        os.system("rm -rf cornell_movie_dialogs_corpus")
        
        print("数据集准备完成")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False
import time
import argparse
import pickle
import numpy as np
from collections import Counter

# 导入模型类
from main import SimpleRNN, clean_text, build_vocab, encode_sentence, DataLoader, train_model

def main():
    # 先下载数据集
    if not download_dataset():
        print("无法下载数据集，训练中止")
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_pairs', type=int, default=10000, help='最大对话对数量')
    args = parser.parse_args()
    
    print("=" * 50)
    print("GitHub Actions 自动训练")
    print("=" * 50)
    print(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, max_pairs={args.max_pairs}")
    print()
    
    # 加载数据
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("错误: 未找到数据文件")
        sys.exit(1)
    
    # 读取对话数据
    conversations = load_conversations(data_dir)
    if len(conversations) > args.max_pairs:
        conversations = conversations[:args.max_pairs]
    print(f"使用 {len(conversations)} 对对话")
    
    # 清洗数据
    cleaned = []
    for src, tgt in conversations:
        src_clean = clean_text(src)
        tgt_clean = clean_text(tgt)
        if src_clean and tgt_clean and len(src_clean.split()) > 1 and len(tgt_clean.split()) > 1:
            cleaned.append((src_clean, tgt_clean))
    print(f"清洗后: {len(cleaned)} 对")
    
    # 构建词表
    all_texts = []
    for src, tgt in cleaned:
        all_texts.append(src)
        all_texts.append(tgt)
    
    vocab, vocab_size = build_vocab(all_texts, min_freq=2, max_vocab=5000)
    idx2word = {v: k for k, v in vocab.items()}
    print(f"词表大小: {vocab_size}")
    
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
    
    # 创建数据加载器
    train_loader = DataLoader(src_seqs_np, tgt_seqs_np, batch_size=args.batch_size)
    
    # 创建模型
    print("创建模型...")
    model = SimpleRNN(vocab_size, embed_size=64, hidden_size=128)
    
    # 训练
    print("开始训练...")
    start_time = time.time()
    
    # 记录训练日志
    log_file = open('training_log.txt', 'w')
    log_file.write("Epoch,Loss,Time\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0
        
        for src, tgt in train_loader:
            src_gpu = np.asarray(src)  # CPU 训练
            tgt_gpu = np.asarray(tgt)
            
            # 前向传播
            logits, hs, embeds = model.forward(src_gpu)
            loss, probs = model.compute_loss(logits, tgt_gpu)
            
            # 反向传播
            model.backward(src_gpu, tgt_gpu, probs, hs, embeds)
            model.update(0.005)
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        log_file.write(f"{epoch+1},{avg_loss:.4f},{epoch_time:.2f}\n")
        log_file.flush()
        
        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            model.save(f'rnn_model_epoch_{epoch+1}.pkl', vocab, idx2word, epoch, avg_loss)
    
    # 保存最终模型
    model.save('rnn_model.pkl', vocab, idx2word, args.epochs-1, avg_loss)
    log_file.close()
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总时间: {total_time/60:.2f} 分钟")
    print(f"最终损失: {avg_loss:.4f}")
    
    # 创建训练报告
    with open('TRAINING_REPORT.md', 'w') as f:
        f.write(f"""# 训练报告

## 训练参数
- 训练轮数: {args.epochs}
- 批次大小: {args.batch_size}
- 数据量: {args.max_pairs} 对话对
- 词表大小: {vocab_size}
- 模型参数: {(vocab_size * 64) + (64 * 128) + (128 * 128) + 128 + (128 * vocab_size) + vocab_size:,} 个

## 训练结果
- 最终损失: {avg_loss:.4f}
- 训练时间: {total_time/60:.2f} 分钟

## 使用方法
1. 下载 `rnn_model.pkl` 文件
2. 放入项目根目录
3. 运行 `python main.py` 选择 "2. Chat only"
""")

def load_conversations(data_dir):
    """加载对话数据"""
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

if __name__ == "__main__":
    main()
