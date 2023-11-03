from cloudevents.http import CloudEvent
from cloudevents.conversion import to_binary
import requests
import time
import os
import shutil
import pickle
import pandas as pd
import numpy as np
import bz2
import sys
from AI import Tokenizer,Model


root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass


#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
vocab_size = 5000
max_len=256


# vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
#             raw_ds,
#             vocabulary_size=vocab_size,
#             reserved_tokens=["[PAD]", "[UNK]", "[BOS]","[EOS]"],
#         )

# with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","wb") as f:
#     pickle.dump(vocab,f)

with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=256,embedding_dim=128,max_len = max_len)

i = 0

while True:
    
    data = os.listdir(log_folder)
    #filtering

    data = [x for x in data if "HDFS" in x]
    data.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder,x)))

    #log rotation and aggregation
    if len(data)>= 64:

        new_logs = []
        new_logs_prep = []
        for d in data[:64]:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read().split("\n")[:-1]

            new_logs += logs

            os.remove(data_path)

    #first step of preprocessing
    parsed_logs += tokenizer.parsing(logs)

    #second step of preprocessing
    vectorized_logs = tokenizer.vectorization(parsed_logs)

    #third step of preprocessing
    enbedded_logs = model.vae.encoder(vectorized_logs)


    new_logs_prep += enbedded_logs
    

        

    compressed_data = bz2.compress(pickle.dumps(new_logs))
    print("lenght of compressed data: ",sys.getsizeof(compressed_data))
    # compressed_data = compressed_data*100
    # print("lenght of compressed data: ",sys.getsizeof(compressed_data))

    headers, _ = to_binary(CloudEvent({
        "type": "logs",
        "source": "simulation",
        "size": str(sys.getsizeof(compressed_data)),
    }, {"data": []}))


    print("sending compressed data:")
    times = []
    for _ in range(100):
        t=time.time()
        r = requests.post("http://reader-service.default:3000",data=compressed_data,headers=headers)
        times.append(time.time()-t)
    print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")


    compressed_data = bz2.compress(pickle.dumps(new_logs_prep))
    print("lenght of encoded compressed data: ",str(sys.getsizeof(compressed_data)))
    # compressed_data = compressed_data*100
    # print("lenght of encoded compressed data: ",str(sys.getsizeof(compressed_data)))

    headers, _ = to_binary(CloudEvent({
        "type": "encoded logs",
        "source": "simulation",
        "size": str(sys.getsizeof(compressed_data)),
    }, {"data": []}))

    print("sending compressed encoded data:")
    times = []
    for _ in range(100):
        t=time.time()
        r = requests.post("http://reader-service.default:3000",data=compressed_data,headers=headers)
        times.append(time.time()-t)
    print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")


    #training step
    metrics = model.train_step(vectorized_logs)
    print(metrics)
        

