import transformers







tokenizer = transformers.AutoTokenizer.from_pretrained("app/bert-base-cased_saved")
# tokens = tokenizer(sentences,return_tensors="np",padding="max_length",max_length=50,return_attention_mask=False,return_token_type_ids =False)["input_ids"]


def preprocess(data):
    tokens = tokenizer(data)["input_ids"]

    return tokens


