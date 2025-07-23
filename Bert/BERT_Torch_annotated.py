'''
BERT模型完整实现 - 带详细中文注释
原作者: Tae Hwan Jung(Jeff Jung) @graykode, 修改: wmathor
参考: https://github.com/jadore801120/attention-is-all-you-need-pytorch
      https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
'''

import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

# ==================== 第一部分：数据预处理 ====================

# 定义训练文本数据 - Romeo和Juliet的对话
text = (
    'Hello, how are you? I am Romeo.\n' # R - Romeo说的话
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J - Juliet说的话
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)

# 数据预处理步骤
# 1. 去除标点符号，转换为小写，按行分割
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # 过滤 '.', ',', '?', '!' 等标点符号

# 2. 构建词汇表
# 将所有句子合并，分割成单词，去重得到词汇列表
word_list = list(set(" ".join(sentences).split())) # ['hello', 'how', 'are', 'you',...]

# 3. 创建词汇映射字典
# 添加BERT特殊token：[PAD]填充, [CLS]分类, [SEP]分隔, [MASK]掩码
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
# 为每个普通单词分配索引（从4开始，因为前4个是特殊token）
for i, w in enumerate(word_list):
    word2idx[w] = i + 4

# 4. 创建反向映射（索引到单词）
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)  # 词汇表大小

# 5. 将每个句子转换为token索引序列
token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]  # 将单词转换为对应的索引
    token_list.append(arr)

# ==================== 第二部分：BERT模型参数配置 ====================

# BERT模型超参数
maxlen = 30          # 最大序列长度
batch_size = 6       # 批次大小
max_pred = 5         # 每个序列最多预测的token数量
n_layers = 6         # Transformer编码器层数
n_heads = 12         # 多头注意力的头数
d_model = 768        # 模型维度
d_ff = 768*4         # 前馈网络维度 (4*d_model)
d_k = d_v = 64       # 注意力机制中K和V的维度
n_segments = 2       # 句子段数（BERT支持两句话）

# ==================== 第三部分：数据生成函数 ====================

def make_data():
    """
    生成BERT训练数据
    包含两个任务：
    1. 掩码语言模型 (Masked Language Model)
    2. 下一句预测 (Next Sentence Prediction)
    """
    batch = []
    positive = negative = 0
    
    # 确保正样本和负样本数量相等
    while positive != batch_size/2 or negative != batch_size/2:
        # 随机选择两个句子
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        
        # 构建BERT输入格式：[CLS] + 句子A + [SEP] + 句子B + [SEP] e.g [1, 36, 25, 34, 22, 6, 38, 7, 13, 11, 39, 26, 19, 2, 35, 21, 23, 2]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        
        # 构建句子段标识：句子A用0，句子B用1 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # ===== 掩码语言模型 (MLM) =====
        # 计算需要掩码的token数量（15%的token）
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))
        
        # 找出可以掩码的位置（排除[CLS]和[SEP]）
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos)
        
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            
            # BERT的掩码策略：
            # 80%概率用[MASK]替换
            # 10%概率用随机词替换
            # 10%概率保持不变
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)
                while index < 4:  # 不能选择特殊token
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index

        # ===== 零填充处理 =====
        # 对输入序列进行填充
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 对掩码token进行填充
        # 确保所有样本的masked_tokens和masked_pos都是固定长度max_pred
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # ===== 下一句预测标签 =====
        # 判断两个句子是否相邻
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    
    return batch

# ==================== 第四部分：数据加载器 ====================

# 生成训练数据
batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)

# 转换为PyTorch张量
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)

class MyDataSet(Data.Dataset):
    """自定义数据集类"""
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]

# 创建数据加载器
loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)

# ==================== 第五部分：BERT模型架构 ====================

def get_attn_pad_mask(seq_q, seq_k):
    """
    生成注意力掩码，防止模型关注到填充token
    """
    batch_size, seq_len = seq_q.size()
    # 找出所有等于0的位置（PAD token）
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
    GELU激活函数实现
    GELU是BERT中使用的激活函数，比ReLU更平滑
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    """BERT的嵌入层：Token嵌入 + 位置嵌入 + 句子段嵌入"""
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token嵌入
        self.pos_embed = nn.Embedding(maxlen, d_model)      # 位置嵌入
        self.seg_embed = nn.Embedding(n_segments, d_model)  # 句子段嵌入
        self.norm = nn.LayerNorm(d_model)                   # 层归一化

    def forward(self, x, seg):
        seq_len = x.size(1)
        # 生成位置索引
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        
        # 三种嵌入相加
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 应用掩码，将填充位置设为极小值
        scores.masked_fill_(attn_mask, -1e9)
        # Softmax归一化
        attn = nn.Softmax(dim=-1)(scores)
        # 计算输出
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 线性变换层
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        
        # 线性变换并重塑为多头
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        # 扩展掩码到多头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算注意力
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # 合并多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # 最终线性变换
        output = nn.Linear(n_heads * d_v, d_model)(context)
        # 残差连接和层归一化
        return nn.LayerNorm(d_model)(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    """位置前馈网络"""
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 两层全连接网络，中间使用GELU激活
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 自注意力 + 前馈网络
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class BERT(nn.Module):
    """完整的BERT模型"""
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
        # 下一句预测的分类头
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        
        # 掩码语言模型的预测头
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # 输出层与嵌入层共享权重（BERT的trick）
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        # 嵌入层
        output = self.embedding(input_ids, segment_ids)
        
        # 生成注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        
        # 通过所有编码器层
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        
        # 下一句预测：使用[CLS]位置的输出
        h_pooled = self.fc(output[:, 0])
        logits_clsf = self.classifier(h_pooled)
        
        # 掩码语言模型：预测被掩码的token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.activ2(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)
        
        return logits_lm, logits_clsf

# ==================== 第六部分：模型训练 ====================

# 初始化模型、损失函数和优化器
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

# 训练循环
import datetime
start = datetime.datetime.now()

for epoch in range(1000):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        # 前向传播
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        
        # 计算损失
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # 掩码语言模型损失
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext)  # 下一句预测损失
        
        # 总损失
        loss = loss_lm + loss_clsf
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end = datetime.datetime.now()
print("训练时间:", end-start)

# ==================== 第七部分：模型预测 ====================

def predict_example():
    """预测示例"""
    # 选择一个样本进行预测
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
    
    print("原始文本:")
    print(text)
    print('=' * 50)
    
    # 显示输入序列
    print("输入序列:")
    print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])
    
    # 模型预测
    logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), 
                                   torch.LongTensor([segment_ids]), 
                                   torch.LongTensor([masked_pos]))
    
    # 获取预测结果
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('真实掩码token:', [pos for pos in masked_tokens if pos != 0])
    print('预测掩码token:', [pos for pos in logits_lm if pos != 0])
    
    # 下一句预测
    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('真实下一句关系:', True if isNext else False)
    print('预测下一句关系:', True if logits_clsf else False)

# 运行预测示例
if __name__ == "__main__":
    predict_example() 