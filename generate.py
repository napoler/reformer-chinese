import argparse
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
import torch
from transformers import *
import os
from reformer_chinese import *
import tkitJson

# pretrained_weights = 'cache/vocab_small_terry_ai.txt'
device='cpu'
output_dir='model'

pretrained_weights=os.path.join(output_dir,'vocab.txt')
config_file=os.path.join(output_dir,'config.json')
Config=tkitJson.Config(config_file)
conf=Config.read()

# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
tokenizer=tokenizer_plus(pretrained_weights)
model = ReformerLM(
    num_tokens= conf['num_tokens'],
    dim = conf['dim'],
    depth = conf['depth'],
    max_seq_len = conf['max_seq_len'],
    lsh_dropout = conf['lsh_dropout'],
    causal = conf['causal'],
    full_attn_thres = conf['full_attn_thres']
)


model_path=os.path.join(output_dir, 'model.pt')

if device=='cuda':
    model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).cuda()
    if os.path.isfile(model_path):
        # if so, load them
        # print('++++'*20)
        model.load_state_dict(torch.load(model_path)).cuda()
else:
    model = TrainingWrapper(model, ignore_index = 0, pad_value = 0).cpu()
    # print(model)
    # print(model.cpu().state_dict())

    # print('++++'*20)
    if os.path.isfile(model_path):
        # if so, load them
        # print('++++'*20)
        print("加载模型")
        model.load_state_dict(torch.load(model_path))
    model.cpu()
    # print(model)
    # print(torch.load(model_path))
    
# model =model.cpu()
# sentence_0 = "你是谁啊"
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
  # print(initial)
  sample = model.generate(initial, length, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
  # print(sample)
  # print(sample.shape) # (1, <=100) token ids
  text = []
  for it in tokenizer.convert_ids_to_tokens(sample.tolist()[0]):
    text.append(it.replace("##",''))

  return text

# get(start_text)


# parser = argparse.ArgumentParser()
# parser.add_argument('--text', default='狗', type=str, required=False, help='设置使用哪些显卡')

# import math
def get_ppl(start_text):
  """
  计算ppl值 语句流畅度
  """
  initial =auto_encode(start_text)
  loss = model(initial, return_loss = True)
  # print(loss)
  loss = loss.mean()
  ppl =torch.exp(loss).item()
  # print(ppl)
  return ppl
  # ppl = math.exp(loss.mean().item())
  # print(ppl)

# args = parser.parse_args()

while True:
  print("\n\n"+"##"*10)
  start_text=input("输入开始词语:")
  pre_text=get(start_text)
  print("".join(pre_text))
  print(get_ppl(start_text+"".join(pre_text)))
