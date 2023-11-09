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
import tensorflow as tf
from AI import Tokenizer,Model


root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass

def compress_and_send(data,type_log,repetitions):

            compressed_data = bz2.compress(pickle.dumps(data))
            print(f"lenght of {type_log}: ", sys.getsizeof(compressed_data))
            
            metrics[type_log].append(sys.getsizeof(compressed_data))

            headers, _ = to_binary(CloudEvent({
                "type": type_log,
                "source": "simulation",
                "size": str(sys.getsizeof(compressed_data)),
            }, {"data": []}))

            times = []
            for _ in range(repetitions):
                t=time.time()
                r = requests.post("http://reader-service.default:3000",data=compressed_data,headers=headers)
                times.append(time.time()-t)
            print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")

#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
vocab_size = 4000
moltiplicatore = 1
max_len=150*moltiplicatore # mean length + std length

with open("./app/logs_tokenizer/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=max_len//3,embedding_dim=128,max_len = max_len)

i = 0
save_iterations = 20

metrics ={"total_loss":[],"reconstruction_loss":[],"kl_loss":[],"logs":[],"vectorized_logs":[],"encoded_logs":[]}

while True:
    
    data = os.listdir(log_folder)
    #filtering

    data = [x for x in data if "HDFS" in x]
    data.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder,x)))

    #log rotation and aggregation
    if len(data)>= 64:
        i += 1
        new_logs = []

        for d in data[:64]:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read().split("\n")[:-1][0]*moltiplicatore

            new_logs.append(logs)

            os.remove(data_path)

        #first step of preprocessing
        parsed_logs = tokenizer.parsing(new_logs)

        #second step of preprocessing
        vectorized_logs = tokenizer.vectorization(parsed_logs)
        # print(vectorized_logs.numpy().shape)
        # print(tf.size(vectorized_logs.numpy()))
        print(f"size of vectorized data: {tf.size(vectorized_logs.numpy()) * vectorized_logs.dtype.size}")

        #third step of preprocessing
        enbedded_logs = model.vae.encode(vectorized_logs)

        print(f"size of enbedded data: {tf.size(enbedded_logs.numpy()) * enbedded_logs.dtype.size}")

        compress_and_send(parsed_logs,"logs",1)
        compress_and_send(vectorized_logs,"vectorized_logs",1)
        compress_and_send(enbedded_logs,"encoded_logs",1)

        #training step
        losses = model.vae.train_step(vectorized_logs)
        print(losses)
        metrics["total_loss"].append(losses["total_loss"].numpy())
        metrics["reconstruction_loss"].append(losses["reconstruction_loss"].numpy())
        metrics["kl_loss"].append(losses["kl_loss"].numpy())

        if i%save_iterations == 0:
            model.vae.save_model(permanent_folder+"/logs_model/")
            with open(permanent_folder+"metrics.pkl","wb") as f:
                pickle.dump(metrics,f)

            

