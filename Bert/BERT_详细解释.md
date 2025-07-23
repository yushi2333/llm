# BERT模型完整实现详解

## 概述

这是一个完整的BERT（Bidirectional Encoder Representations from Transformers）模型实现，使用PyTorch框架。BERT是Google在2018年提出的预训练语言模型，通过双向Transformer编码器来学习文本的上下文表示。

## 代码结构分析

### 第一部分：数据预处理

#### 1. 训练数据
```python
text = (
    'Hello, how are you? I am Romeo.\n' # R - Romeo说的话
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J - Juliet说的话
    # ... 更多对话
)
```
- 使用Romeo和Juliet的对话作为训练数据
- 每行代表一个句子，R表示Romeo，J表示Juliet

#### 2. 文本清洗
```python
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
```
- 去除标点符号：`[.,!?\\-]`
- 转换为小写
- 按行分割成句子列表

#### 3. 词汇表构建
```python
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
```
- 将所有单词去重得到词汇列表
- 添加BERT特殊token：
  - `[PAD]`: 填充token，索引0
  - `[CLS]`: 分类token，索引1
  - `[SEP]`: 分隔token，索引2
  - `[MASK]`: 掩码token，索引3

#### 4. Token化
```python
token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)
```
- 将每个句子转换为token索引序列

### 第二部分：模型参数配置

```python
maxlen = 30          # 最大序列长度
batch_size = 6       # 批次大小
max_pred = 5         # 每个序列最多预测的token数量
n_layers = 6         # Transformer编码器层数
n_heads = 12         # 多头注意力的头数
d_model = 768        # 模型维度
d_ff = 768*4         # 前馈网络维度
d_k = d_v = 64       # 注意力机制中K和V的维度
n_segments = 2       # 句子段数
```

### 第三部分：数据生成函数

#### make_data()函数详解

这个函数生成BERT的两个预训练任务的数据：

##### 1. 掩码语言模型 (Masked Language Model, MLM)
```python
# 计算需要掩码的token数量（15%的token）
n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))

# BERT的掩码策略：
if random() < 0.8:  # 80%概率用[MASK]替换
    input_ids[pos] = word2idx['[MASK]']
elif random() > 0.9:  # 10%概率用随机词替换
    index = randint(0, vocab_size - 1)
    input_ids[pos] = index
# 10%概率保持不变
```

**掩码策略说明：**
- 80%概率：用`[MASK]`替换原token
- 10%概率：用随机词替换
- 10%概率：保持原token不变

这种策略的目的是让模型学会从上下文中预测被掩盖的词。

##### 2. 下一句预测 (Next Sentence Prediction, NSP)
```python
if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
    batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
    batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
```

**NSP任务说明：**
- 正样本：两个句子在原文中相邻
- 负样本：两个句子在原文中不相邻
- 模型需要判断第二个句子是否是第一个句子的下一句

### 第四部分：BERT模型架构

#### 1. 嵌入层 (Embedding)
```python
class Embedding(nn.Module):
    def __init__(self):
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token嵌入
        self.pos_embed = nn.Embedding(maxlen, d_model)      # 位置嵌入
        self.seg_embed = nn.Embedding(n_segments, d_model)  # 句子段嵌入
```

**三种嵌入：**
- **Token嵌入**：将token ID转换为向量表示
- **位置嵌入**：编码token在序列中的位置信息
- **句子段嵌入**：区分不同的句子（0表示第一句，1表示第二句）

#### 2. 缩放点积注意力 (Scaled Dot-Product Attention)
```python
def forward(self, Q, K, V, attn_mask):
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
    scores.masked_fill_(attn_mask, -1e9)
    attn = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn, V)
```

**注意力机制步骤：**
1. 计算Query和Key的点积
2. 除以缩放因子`√d_k`
3. 应用掩码（将填充位置设为极小值）
4. Softmax归一化
5. 与Value相乘得到输出

#### 3. 多头注意力 (Multi-Head Attention)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
```

**多头注意力机制：**
- 将输入分别投影到多个子空间
- 在每个子空间中计算注意力
- 将所有头的输出拼接后投影回原维度
- 使用残差连接和层归一化

#### 4. 位置前馈网络 (Position-wise Feed-Forward Network)
```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
```

**前馈网络：**
- 两层全连接网络
- 中间使用GELU激活函数
- 输入输出维度相同

#### 5. 编码器层 (Encoder Layer)
```python
class EncoderLayer(nn.Module):
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs
```

**编码器层结构：**
- 多头自注意力 + 残差连接 + 层归一化
- 位置前馈网络 + 残差连接 + 层归一化

#### 6. 完整BERT模型
```python
class BERT(nn.Module):
    def __init__(self):
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # 两个任务头
        self.classifier = nn.Linear(d_model, 2)  # NSP任务
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)  # MLM任务
```

**BERT模型特点：**
- 多层Transformer编码器
- 两个预训练任务头
- MLM任务的输出层与嵌入层共享权重（BERT的trick）

### 第五部分：训练过程

#### 损失函数
```python
loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # MLM损失
loss_clsf = criterion(logits_clsf, isNext)  # NSP损失
loss = loss_lm + loss_clsf  # 总损失
```

**训练目标：**
- **MLM损失**：预测被掩码的token
- **NSP损失**：判断两个句子是否相邻
- **总损失**：两个损失的和

#### 训练循环
```python
for epoch in range(1000):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss = loss_lm + loss_clsf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 第六部分：模型预测

#### 预测示例
```python
def predict_example():
    # 选择一个样本进行预测
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
    
    # 模型预测
    logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), 
                                   torch.LongTensor([segment_ids]), 
                                   torch.LongTensor([masked_pos]))
    
    # 获取预测结果
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
```

## BERT的核心创新

### 1. 双向编码
- 传统语言模型只能从左到右或从右到左编码
- BERT使用双向Transformer，能同时看到上下文

### 2. 预训练任务
- **MLM**：让模型学会理解词汇的上下文含义
- **NSP**：让模型学会理解句子间的关系

### 3. 通用表示
- 预训练后的BERT可以用于多种下游任务
- 通过微调适应特定任务

## 应用场景

1. **文本分类**：使用[CLS]位置的输出
2. **命名实体识别**：使用每个位置的输出
3. **问答系统**：使用两个句子作为输入
4. **文本相似度**：使用句子对的表示

## 总结

这个实现展示了BERT的核心组件：
- 数据预处理和token化
- 两个预训练任务的数据生成
- 完整的Transformer编码器架构
- 多头注意力机制
- 训练和预测流程

通过这个实现，可以深入理解BERT的工作原理和实现细节。 