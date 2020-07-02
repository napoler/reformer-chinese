from transformers import *
from reformer_chinese import *

# def tokenizer_plus(vocab='cache/vocab_small_terry_ai.txt'):
#     full_tokenizer = BertTokenizer.from_pretrained(vocab)
#     vocab_list=list(full_tokenizer.get_vocab())
#     full_tokenizer=BertTokenizer(vocab,never_split=vocab_list)
#     return full_tokenizer

full_tokenizer=tokenizer_plus()
line=" [KW] 皇帝专业户张铁林，竟有如此青涩稚嫩的模样 [/KW]  [TT]皇帝专业户张铁林， [SEP]  竟有如此青涩稚嫩的模样 [/TT]  [CLS] "
ids=full_tokenizer.tokenize(line) 

ids=full_tokenizer.convert_tokens_to_ids(ids)
print(ids)

print(full_tokenizer.convert_tokens_to_ids('[KW]'))