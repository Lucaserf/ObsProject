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

def compress_and_send(data,type_log,repetitions,i,catching_time):
            
            compressed_data = bz2.compress(pickle.dumps(data))
            print(f"lenght of {type_log}: ", sys.getsizeof(compressed_data))
            
            metrics[type_log].append(sys.getsizeof(compressed_data))

            headers, _ = to_binary(CloudEvent({
                "id": str(i),
                "type": type_log,
                "source": "simulation",
                "time": str(catching_time),
            }, {"data": []}))

            times = []
            for _ in range(repetitions):
                t=time.time()
                r = requests.post("http://reader-service.default:3000",data=compressed_data,headers=headers)
                times.append(time.time()-t)
            print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")

#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
vocab_size = 5000
max_len=85 # mean length + std length
latent_dim=max_len//2
threshold = 530

with open("./app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/15")

i = 0
save_iterations = 20

metrics ={"total_loss":[],"reconstruction_loss":[],"kl_loss":[],"logs":[],"vectorized_logs":[],"encoded_logs":[],"mean_padding":[],"anomaly":[]}

while True:
    
    data = os.listdir(log_folder)
    #filtering

    data = [x for x in data if "BGL" in x]
    data.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder,x)))

    #log rotation and aggregation
    if len(data)>= 1:
        i += 1
        new_logs = []

        for d in data[:1]:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read().split("\n")[0]

            new_logs.append(logs)

            os.remove(data_path)

        new_logs = tf.constant(new_logs)

        log_catch_time = time.time()

        # #first step of preprocessing
        # parsed_logs = tokenizer.parsing(new_logs)

        # time_after_parse = time.time()

        
        vectorized_logs = tokenizer.vectorization(new_logs)

        # time_after_vectorization = time.time()

        # loss = model.vae.get_loss(vectorized_logs)

        
        # anomaly = False
        # if loss > threshold:
        #     anomaly = True

        # time_after_detection = time.time()

        # compress_and_send(new_logs,"logs",1,i,log_catch_time)
        compress_and_send(vectorized_logs,"vectorized_logs",1,i,log_catch_time)
        # compress_and_send(anomaly,"anomaly",1,i,time_after_detection)
        
        #training step
        # metrics["mean_padding"].append(tf.reduce_mean(tf.reduce_sum(tf.cast(vectorized_logs==0,tf.int32),axis=-1)).numpy())
        # metrics["reconstruction_loss"].append(loss)

        #saving model after a number of iterations
        # if i%save_iterations == 0:
        #     # model.vae.save_model(permanent_folder+"/logs_model/")
        #     with open(permanent_folder+"metrics.pkl","wb") as f:
        #         pickle.dump(metrics,f)

        


        

            

