import torch
from reformer_pytorch import ReformerEncDec
import tkitJson
from tkitMatch import Match
import argparse
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
import torch
from transformers import *
import os
from reformer_chinese import *
import tkitJson
from tkitMatch import Match


# pretrained_weights = 'cache/vocab_small_terry_ai.txt'
device='cuda'
output_dir='model'

pretrained_weights=os.path.join(output_dir,'vocab.txt')
# config_file=os.path.join(output_dir,'config.json')
# Config=tkitJson.Config(config_file)
# conf=Config.read()

# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
tokenizer=tokenizer_plus(pretrained_weights)
full_tokenizer=tokenizer

model_path=os.path.join(output_dir, 'model.pt')



def auto_encode(sentence_0):
  # sentence_1 = "我是谁啊"
    sentence_1=None
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=False, return_tensors='pt')
    # inputs_1=tokenizer.convert_tokens_to_ids(sentence_0)
    # inputs_1 = torch.tensor(inputs_1).long()
    return inputs_1['input_ids']



DE_SEQ_LEN = 256
EN_SEQ_LEN = 256

model = ReformerEncDec(
    dim = 128, 
    enc_num_tokens = full_tokenizer.vocab_size,
    enc_depth = 6,
    enc_max_seq_len =EN_SEQ_LEN ,
    dec_num_tokens =full_tokenizer.vocab_size,
    dec_depth = 12,
    dec_max_seq_len = DE_SEQ_LEN
)

model.load_state_dict(torch.load(model_path))
model.to(device)
# start_text=input("输入带提取的句子：")
Tjson=tkitJson.Json("data/train.json")
for item in Tjson.load():
    print("#"*10)
    start_text=item["sentenceA"]
    sentA_ids=full_tokenizer.encode_plus(start_text,max_length=EN_SEQ_LEN,pad_to_max_length=True)['input_ids']
    # evaluate with the following
    # eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long()
    eval_seq_in = torch.tensor([sentA_ids]).long().to(device)
    eval_seq_out_start = torch.tensor([[0.]]).long().to(device) # assume 0 is id of start token


    # 定义输出开始词语
    out_start_text=item["sentenceB"].split(" [KGS]")[0]
    # out_start_text="[NER]"
    eval_seq_out_start=full_tokenizer.encode_plus(out_start_text,pad_to_max_length=False)['input_ids'][:-1]
    # print(eval_seq_out_start)
    eval_seq_out_start = torch.tensor([eval_seq_out_start]).long().to(device) # assume 0 is id of start token
    print("--"*20)
    print(item)
    # print(eval_seq_in)
    # print(eval_seq_out_start)
    samples = model.generate(eval_seq_in, eval_seq_out_start, seq_len = DE_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
    # print(samples)
    # print(samples.shape) # (1, <= 1024) decode the tokens
    

    text=[]
    for it in tokenizer.convert_ids_to_tokens(samples.tolist()[0]):

        if it=="[PAD]":
            continue
        elif it=="[/KGS]":
            text.append(it.replace("##",''))
            break
        text.append(it.replace("##",''))
    print("".join(text))
