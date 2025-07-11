import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import Transformer
from batch import create_masks
from process import *
import numpy as np
import time

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据
src_file = 'data/english.txt'
trg_file = 'data/french.txt'
src_lang = 'en_core_web_sm'
trg_lang = 'fr_core_news_sm'
max_strlen = 80
batchsize = 1500
src_data, trg_data = read_data(src_file, trg_file)  # 一个包含所有源语言（英语）句子的字符串列表。154883
EN_TEXT, FR_TEXT = create_fields(src_lang, trg_lang)
train_iter, src_pad, trg_pad = create_dataset(src_data, trg_data, EN_TEXT, FR_TEXT, max_strlen, batchsize) # , 1 , 1

'''1.1.5 编码器和解码器结构'''
# 模型参数定义
d_model = 512
heads = 8
N = 6
dropout = 0.1
src_vocab = len(EN_TEXT.vocab) #13724
trg_vocab = len(FR_TEXT.vocab) #23469
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, dropout)
model.to(device) # Move model to device

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 模型训练
def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.src.transpose(0, 1).to(device) # Move src to device
            trg = batch.trg.transpose(0, 1).to(device) # Move trg to device (107*14)
            # 将我们输入的英语句子中的所有单词翻译成法语
            # 除了最后一个单词，因为它为结束符，不需要进行下一个单词的预测

            trg_input = trg[:, :-1] # trg_input is already on device as it's a slice of trg

            # 试图预测单词 这一变量用于计算损失，其中每一句目标语句都删去起始符<sos>
            targets = trg[:, 1:].contiguous().view(-1).to(device) # Move targets to device

            # 使用掩码代码创建函数来制作掩码
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            src_mask = src_mask.to(device) # Move src_mask to device
            trg_mask = trg_mask.to(device) # Move trg_mask to device

            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                   targets, ignore_index=trg_pad)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % 
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, 
                       time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()

def entosentence(src):
    # 假设 src[0] 是您在 PDB 中看到的 tensor
    # 假设 EN_TEXT 和 src_pad 变量在当前作用域可访问

    # 1. 将 Tensor 移动到 CPU 并转换为 Python 列表
    index_list = src.cpu().tolist()

    # 2. 获取词汇表的 itos (index to string) 映射
    itos = EN_TEXT.vocab.itos

    # 3. 获取 padding 的索引
    pad_index = src_pad # 或者直接使用 EN_TEXT.vocab.stoi['<pad>']

    # 4. 遍历索引列表，查找单词，并过滤掉 padding
    words = [itos[index] for index in index_list if index != pad_index]

    # 5. 将单词连接成句子
    sentence = ' '.join(words)

    print(sentence)

def frtosentence(src):
    # 假设 src[0] 是您在 PDB 中看到的 tensor
    # 假设 EN_TEXT 和 src_pad 变量在当前作用域可访问

    # 1. 将 Tensor 移动到 CPU 并转换为 Python 列表
    index_list = src.cpu().tolist()

    # 2. 获取词汇表的 itos (index to string) 映射
    itos = FR_TEXT.vocab.itos

    # 3. 获取 padding 的索引
    pad_index = trg_pad # 或者直接使用 EN_TEXT.vocab.stoi['<pad>']

    # 4. 遍历索引列表，查找单词，并过滤掉 padding
    words = [itos[index] for index in index_list if index != pad_index]

    # 5. 将单词连接成句子
    sentence = ' '.join(words)

    print(sentence)

# 模型测试
def translate(src, max_len=80, custom_string=False):
    model.eval()
    if custom_string == True:
        src = tokenize_en(src, EN_TEXT)
        src = torch.LongTensor(src)
    print(src)
    src = src.to(device) # Move input src to device
    src_mask = (src != src_pad).unsqueeze(-2).to(device) # Move src_mask to device
    e_outputs = model.encoder(src.unsqueeze(0), src_mask) #(1,4,512)

    outputs = torch.zeros(max_len).type_as(src.data).to(device) # Create outputs on device (80)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']]).to(device) # Move sos token to device

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)).astype('uint8'))
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device) # Move trg_mask to device

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), 
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join(
        [FR_TEXT.vocab.itos[ix] for ix in outputs[:i]]
    )

if __name__ == "__main__":
    train_model(500)

    words = 'Let me see.'
    print(translate(words, custom_string=True))