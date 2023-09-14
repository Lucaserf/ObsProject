import transformers
import numpy as np
import pickle
import re


with open("../../../docker_app/app/quotes.txt") as f:
    list_of_quotes = f.read()

print(list_of_quotes[:500])
list_of_quotes = re.sub(r"\n--(.*)\n\n",r"\1\t",list_of_quotes)
list_of_quotes = re.sub(r"\n",r" ",list_of_quotes)

list_of_quotes = re.split(r"\t",list_of_quotes)

print(list_of_quotes[:5])




# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
# # tokens = tokenizer(sentences,return_tensors="np",padding="max_length",max_length=50,return_attention_mask=False,return_token_type_ids =False)["input_ids"]

# tokens = tokenizer(sentences)["input_ids"]


# print(tokens)

# with open("data.log","w") as f:
#     f.write("\n".join(sentences))

# with open("encoded_data.log","wb") as f:
#     pickle.dump(tokens,f)

# with open("encoded_data.log","rb") as f:
#     data = pickle.load(f)

# print(data)