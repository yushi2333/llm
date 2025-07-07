import pandas as pd
import torchtext
from torchtext import data
from tokenizer import tokenize
from batch import MyIterator, batch_size_fn
import os

# 读取源语言和目标语言的数据文件，每行为一句话
def read_data(src_file, trg_file):
    if src_file is not None:
        try:
            # 读取源文件内容，去除首尾空白并按行分割
            src_data = open(src_file).read().strip().split('\n')
        except:
            # 文件未找到时输出错误信息并退出
            print("error: '" + src_file + "' file not found")
            quit()
    if trg_file is not None:
        try:
            # 读取目标文件内容，去除首尾空白并按行分割
            trg_data = open(trg_file).read().strip().split('\n')
        except:
            print("error: '" + trg_file + "' file not found")
            quit()
    # 返回两个列表，分别为源数据和目标数据
    return src_data, trg_data

# 创建用于分词和数值化的Field对象
def create_fields(src_lang, trg_lang):
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    # 检查源语言和目标语言是否在支持列表中
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + str(spacy_langs))
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + str(spacy_langs))
    
    print("loading spacy tokenizers...")
    # 获取分词器
    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    # 定义目标语言和源语言的Field对象
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    # 返回Field对象
    return(SRC, TRG)

# 创建数据集和迭代器
def create_dataset(src_data, trg_data, SRC, TRG, max_strlen, batchsize):
    print("creating dataset and iterator... ")

    # 构造DataFrame，包含源句子和目标句子
    raw_data = {'src' : [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    # 过滤掉长度超过max_strlen的句子
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    # 保存为临时csv文件
    df.to_csv("translate_transformer_temp.csv", index=False)
    
    # 定义字段
    data_fields = [('src', SRC), ('trg', TRG)]
    # 读取csv文件，创建TabularDataset
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    # 创建自定义迭代器
    train_iter = MyIterator(train, batch_size=batchsize, device='cpu',
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    # 构建词表
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # 获取pad的索引
    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    # 说明：pad填充由torchtext的Field自动完成。
    # 在每个batch内部，会自动将每句话pad到本batch中最长句子的长度，pad内容为<pad>。
    # pad的索引可通过SRC.vocab.stoi['<pad>']获取，后续可用于mask。

    # 获取训练集的批次数
    train_len = get_len(train_iter)
    print(train_len)

    # 返回迭代器和pad索引
    return train_iter, src_pad, trg_pad

# 计算迭代器中的批次数
def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

# 对英文句子分词并转换为索引
def tokenize_en(src, SRC):
    spacy_en = tokenize('en_core_web_sm')
    src = spacy_en.tokenizer(src)
    # 返回每个分词在词表中的索引
    return [SRC.vocab.stoi[tok] for tok in src]