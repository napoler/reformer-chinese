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
import re 


# def filter_tags(htmlstr):
#     #先过滤CDATA
#     re_cdata=re.compile('//<![CDATA[[^>]*//]]>',re.I) #匹配CDATA
#     re_script=re.compile('<s*script[^>]*>[^<]*<s*/s*scripts*>',re.I)#Script
#     re_style=re.compile('<s*style[^>]*>[^<]*<s*/s*styles*>',re.I)#style
#     re_kg=re.compile('[s*KG[^>]*][^<]*[s*/s*KG*]',re.I)#style
#     re_br=re.compile('<brs*?/?>')#处理换行
#     re_h=re.compile('</?w+[^>]*>')#HTML标签
#     re_comment=re.compile('<!--[^>]*-->')#HTML注释
#     s=re_cdata.sub('',htmlstr)#去掉CDATA
#     s=re_script.sub('',s) #去掉SCRIPT
#     s=re_kg.sub('',s)#re_kg
#     s=re_style.sub('',s)#去掉style
#     s=re_br.sub('n',s)#将br转换为换行
#     s=re_h.sub('',s) #去掉HTML 标签
#     s=re_comment.sub('',s)#去掉HTML注释
#     #去掉多余的空行
#     blank_line=re.compile('n+')
#     s=blank_line.sub('n',s)
#     s=replaceCharEntity(s)#替换实体
#     return s


def filter_tags(texts):
  reObj1 = re.compile('((KG)\s+KG)') 
  print(reObj1.findall(texts[0]) )

  # pattern = re.compile('KG')
  # for text in texts:
  #   print(pattern.search(text))
    # return s
def get_kg(start_text,length=50):
  """
  获取预测知识
  """
  data=[]
  # for w in get(start_text,length):
  #   # print(w)
  #   if w=='[/KGS]':
  #     break
  #   elif w=='[/KG]':
  #     print("".join(words))
  #     # words=[]
  #     data.append(words)
  #     pass
  #   elif w=='[KG]':
  #     words=[]
  #     pass
  #   elif w=='[S]':
  #     pass
  #   elif len(words)>=0:
  #     words.append(w)
  text=get(start_text,length)
  print("".join(text))
  data = filter_tags(["".join(text)])
  return data

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
file="data/train.txt"
fp=open(file,'r')
lines = fp.readlines()
for line in lines:
  print("\n"*3)
  print("###"*20)
  print("原始语句：",line.split("[KGS]")[0])
  try:
    print("正确知识：",line.split("[KGS]")[1])
  except:
    pass
  # print(line.split("[KGS]"))
  start_text=line.split("[KGS]")[0]+" [KGS] "
  print(get_kg(start_text))
  # pre_text=get(start_text)
  # p="".join(pre_text)
  # # p.split("[/KGS]")[0]
  # print("预测结果",p.split("[/KGS]")[0])
  # # print(get_ppl(start_text+"".join(pre_text)))
