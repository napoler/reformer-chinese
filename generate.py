import argparse
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
import torch
from transformers import *
import os
pretrained_weights = 'cache/vocab_small_terry_ai.txt'
device='cpu'
output_dir='model'


tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = ReformerLM(
    num_tokens= 13137,
    dim = 1024,
    depth = 12,
    max_seq_len = 4096,
    lsh_dropout = 0.1,
    causal = True,
    full_attn_thres = 1024
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



def get(start_text,length=30):
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
  return text

# get(start_text)


# parser = argparse.ArgumentParser()
# parser.add_argument('--text', default='狗', type=str, required=False, help='设置使用哪些显卡')


# args = parser.parse_args()

while True:
    start_text=input("输入开始词语:")
    pre_text=get(start_text)
    print("".join(pre_text))
