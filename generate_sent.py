import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
# from tokenizations.bpe_tokenizer import get_encoder
# import pre_process_data as ppd
import pickle
from transformers import *
import torch
import os
from torch import randint
import torch.nn as nn
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
from transformers import BertTokenizer, AdamW
# from processors import *
import pickle
import tkitFile 
from reformer_chinese import *

pretrained_weights = 'cache/vocab_small_terry_ai.txt'
tokenizer=tokenizer_plus(pretrained_weights)
tokenizer.max_len = 128
def get_data(path,tokenizer):
    
    # temp=tkitFile.Json('data/cache/train.json')
    for it in tkitFile.Json(path).auto_load():
        item={}
        # print(it)

        kw = tokenizer.encode_plus(it['keywords'], max_length=tokenizer.max_len, add_special_tokens=True)
        pad_num=tokenizer.max_len-len(kw['input_ids'])
        item['keywords']=kw['input_ids']+ [tokenizer.convert_tokens_to_ids('[PAD]')]*pad_num

        tx= tokenizer.encode_plus(it['text'], max_length=tokenizer.max_len, add_special_tokens=True) 
        pad_num=tokenizer.max_len-len(tx['input_ids'])
        item['text']=tx['input_ids']+ [tokenizer.convert_tokens_to_ids('[PAD]')]*pad_num

        #   inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
        # print(item)
        yield item



def auto_encode(sentence_0):
  # sentence_1 = "我是谁啊"
    sentence_1=None
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=False, return_tensors='pt')
    # inputs_1=tokenizer.convert_tokens_to_ids(sentence_0)
    # inputs_1 = torch.tensor(inputs_1).long()
    return inputs_1['input_ids']



def get(start_text,length=50):
  """
  获取预测文本
  """
  # start_text=x_train_text[0][:5]
  initial =auto_encode(start_text)
#   print(initial)
  sample = model.generate(initial, length, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
#   print(sample)
  # print(sample.shape) # (1, <=100) token ids
  text = tokenizer.convert_ids_to_tokens(sample.tolist()[0])

            #   if multi_gpu:
            #     loss = loss.mean()
            # total_loss += loss
            # total_steps += 1

            # if (overall_step + 1) % log_step == 0:
            #     print('now time: {}:{}. Step {} of piece {}, ppl {}'.format(
            #         datetime.now().hour,
            #         datetime.now().minute,
            #         (step + 1),
            #         piece_num,
            #         torch.exp(loss)))
  return text

def gen(text):
    model = ReformerLM(
        num_tokens= 13137,
        dim = 128,
        depth = 12,
        max_seq_len = 4096,
        lsh_dropout = 0.1,
        causal = True,
        full_attn_thres = 128
    )
    model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).cpu()
    output_dir="model"
    model_cpu_path=os.path.join(output_dir, 'model_cpu.pt')
    model.load_state_dict(torch.load(model_cpu_path))
    initial =auto_encode(text)
    #   print(initial)
    sample = model.generate(initial, 10, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
    #   print(sample)
    # print(sample.shape) # (1, <=100) token ids
    text = tokenizer.convert_ids_to_tokens(sample.tolist()[0])
    # print(text)
    return text




def train(device='cpu',output_dir='model',epochs=5,save_step=5,batch_size=4):

    model = ReformerLM(
        num_tokens= 13137,
        dim = 128,
        depth = 12,
        max_seq_len = 4096,
        lsh_dropout = 0.1,
        causal = True,
        full_attn_thres = 128
    )
    model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).to(device)
    # output_dir="model"
    model_cpu_path=os.path.join(output_dir, 'model_cpu.pt')
    try:
        model.load_state_dict(torch.load(model_cpu_path))
    except:
        pass

    model.train()
    optimizer = AdamW(params=model.parameters())
    optimizer_path=os.path.join(output_dir, 'optimizer.pt')
    try:
        optimizer.load_state_dict(torch.load(optimizer_path))
    except:
        pass
    print(optimizer)
    total_loss = 0.0
    # batch_size=4

    loss = []

    data=[]
    for it in get_data("data/train.json",tokenizer):
        data.append(it)
    # data=data[:1000]
    loss_fn = nn.CrossEntropyLoss()  # -100 index = padding token
    for n in tqdm(range(epochs)):
        # print(n)
        random.shuffle(data)
        inputs=[]
        labels=[]
        for i,it in enumerate( data):
            # print("it",it)
            inputs.append(it['keywords'])
            labels.append(it['text'])
            if i %batch_size==0 and i!=0:
                # print(it)

                inputs_batch = torch.tensor(inputs).long().to(device)


                labels_batch = torch.tensor(labels).long().to(device)
                # print(inputs_batch)
                inputs=[]
                labels=[]


                # inputs = torch.tensor(it['keywords']).long()
                # labels = torch.tensor(it['text']).long()
                # print("inputs",inputs)
                pred = model(inputs_batch)
                mlm_loss = loss_fn(pred.view(-1, tokenizer.vocab_size), labels_batch.view(-1))
                
                total_loss += mlm_loss.item()
                loss.append(mlm_loss.item())
                print('loss',mlm_loss.item())
                mlm_loss.backward()
                optimizer.step()
                model.zero_grad()
                # output_dir="model"
            if i% save_step==0 and i!=0:
                model_cpu_path=os.path.join(output_dir, 'model_cpu.pt')
                optimizer_path=os.path.join(output_dir, 'optimizer.pt')
                torch.save(model.state_dict(), model_cpu_path)
                torch.save(optimizer.state_dict(), optimizer_path)
        model_cpu_path=os.path.join(output_dir, 'model_cpu.pt')
        optimizer_path=os.path.join(output_dir, 'optimizer.pt')
        torch.save(model.state_dict(), model_cpu_path)
        torch.save(optimizer.state_dict(), optimizer_path)
# train()
# print("1:训练 2:生成")
# x=input("选择：")
# if x =='1':
#     train()
# else:
x=input("关键词")
words=gen(x)
print("".join(words))
