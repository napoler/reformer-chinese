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

import tkitFile 


tokenizer = BertTokenizer.from_pretrained('cache/vocab_small_terry_ai.txt')
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
    print(text)




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
#     x=input("关键词")
#     gen(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, required=False, help='设置使用哪些显卡')
    # parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
    #                     help='选择模型参数')
    # parser.add_argument('--tokenizer_path', default='cache/vocab_small_terry_ai.txt', type=str, required=False, help='选择词库')
    # parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    # parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        # help='tokenized语料存放位置')
    # parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-8, type=float, required=False, help='学习率')
    # parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    # parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    # parser.add_argument('--stride', default=1024, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=5, type=int, required=False, help='梯度积累')
    # parser.add_argument('--fp16', action='store_true', help='混合精度')
    # parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    # parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    # parser.add_argument('--num_pieces', default=10, type=int, required=False, help='将训练语料分成多少份')
    # parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    # parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    # parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    # parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    # parser.add_argument('--bpe_token', action='store_true', help='subword')
    # parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    # parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args',args)
    train(device=args.device,output_dir=args.output_dir,epochs=args.epochs,save_step=5,batch_size=args.batch_size)

if __name__ == '__main__':
    main()
