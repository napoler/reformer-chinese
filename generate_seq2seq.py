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
device='cpu'
output_dir='model'

pretrained_weights=os.path.join(output_dir,'vocab.txt')
# config_file=os.path.join(output_dir,'config.json')
# Config=tkitJson.Config(config_file)
# conf=Config.read()

# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
tokenizer=tokenizer_plus(pretrained_weights)
full_tokenizer=tokenizer

model_path=os.path.join(output_dir, 'model.pt')

# if device=='cuda':
#     model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).cuda()
#     if os.path.isfile(model_path):
#         # if so, load them
#         # print('++++'*20)
#         model.load_state_dict(torch.load(model_path)).cuda()
# else:
#     model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).cpu()










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
    dim = 256, 
    enc_num_tokens = full_tokenizer.vocab_size,
    enc_depth = 12,
    enc_max_seq_len = DE_SEQ_LEN,
    dec_num_tokens =full_tokenizer.vocab_size,
    dec_depth = 12,
    dec_max_seq_len = EN_SEQ_LEN
)

model.load_state_dict(torch.load(model_path))
model.to("cuda")
start_text=input("输入带提取的句子：")
sentA_ids=full_tokenizer.encode_plus(start_text,max_length=EN_SEQ_LEN,pad_to_max_length=True)['input_ids']
# evaluate with the following
# eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long()
eval_seq_in = torch.tensor([sentA_ids]).long().to("cuda")
eval_seq_out_start = torch.tensor([[0.]]).long().to("cuda") # assume 0 is id of start token
print(eval_seq_in)
print(eval_seq_out_start)
samples = model.generate(eval_seq_in, eval_seq_out_start, seq_len = DE_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
print(samples)
print(samples.shape) # (1, <= 1024) decode the tokens


text=[]
for it in tokenizer.convert_ids_to_tokens(samples.tolist()[0]):
    text.append(it.replace("##",''))
print("".join(text))
