import torch
from reformer_pytorch import ReformerEncDec

DE_SEQ_LEN = 256
EN_SEQ_LEN = 256

enc_dec = ReformerEncDec(
    dim = 128,
    enc_num_tokens = 20000,
    enc_depth = 6,
    enc_max_seq_len = DE_SEQ_LEN,
    dec_num_tokens = 20000,
    dec_depth = 6,
    dec_max_seq_len = EN_SEQ_LEN
)

train_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long()
train_seq_out = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long()
input_mask = torch.ones(1, DE_SEQ_LEN).bool()

loss = enc_dec(train_seq_in, train_seq_out, return_loss = True, enc_input_mask = input_mask)
loss.backward()
# learn

# evaluate with the following
eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long()
eval_seq_out_start = torch.tensor([[0.]]).long() # assume 0 is id of start token
print(eval_seq_in)
print(eval_seq_out_start)
samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len = EN_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
print(samples)
print(samples.shape) # (1, <= 1024) decode the tokens