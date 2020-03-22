def tokenizer_plus(vocab='cache/vocab_small_terry_ai.txt'):
    full_tokenizer = BertTokenizer.from_pretrained(vocab)
    vocab_list=list(full_tokenizer.get_vocab())
    full_tokenizer=BertTokenizer(vocab,never_split=vocab_list)
    return full_tokenizer
