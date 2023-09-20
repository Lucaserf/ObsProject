
import transformers



class Tokenizer():
    def __init__(self,path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path,max_lenth = 512)
        # tokens = tokenizer(sentences,return_tensors="np",padding="max_length",max_length=50,return_attention_mask=False,return_token_type_ids =False)["input_ids"]

    def training_tokenizer(self,training_corpus):
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, 32000)

    def saving_tokenizer(self,path):
        self.tokenizer.save_pretrained(path)

    def preprocess(self,data):
        tokens = self.tokenizer(data)["input_ids"]

        return tokens


    def decode(self,data):
        decoded_data = self.tokenizer.batch_decode(data)
        return decoded_data

