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
model_cpu_path=os.path.join(output_dir, 'model_cpu.pt')
torch.save(model.cpu().state_dict(), model_cpu_path)