# from cloudevents.http import CloudEvent
# from cloudevents.conversion import to_binary
# import requests
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
import zmq
import time

root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://reader-service.default:3000")

operation_mode = os.environ["OPERATION_MODE"]



def compress_and_send(data,type_log,i,log_creation_time,catching_time,after_preprocess_time):
            
            # compressed_data = bz2.compress(pickle.dumps(data))
            # print(f"lenght of {type_log}: ", sys.getsizeof(compressed_data))
            
            # metrics[type_log].append(sys.getsizeof(compressed_data))

            # headers, _ = to_binary(CloudEvent({
            #     "id": str(i),
            #     "type": type_log,
            #     "source": "simulation",
            #     "time": str(catching_time),
            # }, {"data": []}))
            compressed_data = bz2.compress(pickle.dumps(data))

            event = {
                "id_node": os.environ["HOSTNAME"].split("-")[-1],
                "id": i,
                "type": type_log,
                "source": "simulation",
                "catch_time": catching_time,
                "after_preprocess_time": after_preprocess_time,
                "log_creation_time": log_creation_time,
                "data": compressed_data,
                "data_size" : sys.getsizeof(compressed_data)
            }

            compressed_event = bz2.compress(pickle.dumps(event))
            

            socket.send(compressed_event)
            # r = requests.post("http://reader-service.default:3000",data=compressed_data,headers=headers)


#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
vocab_size = 5000
max_len=85 # mean length + std length
latent_dim=max_len//2
threshold = 530
number_logs_to_send = 1000

with open("./app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/15")

i = 0
save_iterations = 20

metrics ={"total_loss":[],"reconstruction_loss":[],"kl_loss":[],"logs":[],"vectorized_logs":[],"encoded_logs":[],"mean_padding":[],"anomaly":[]}

time_last_send = time.time()

while True:  #i< number_logs_to_send:
    
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
        
            log_creation_time = float(d[3:-4])


        new_logs = tf.constant(new_logs)


        log_catch_time = time.time()

        if operation_mode == "logs":
            output = new_logs

        elif operation_mode == "vectorized_logs":
            output = tokenizer.vectorization(new_logs)

        elif operation_mode == "anomaly":  
            vectorized_logs = tokenizer.vectorization(new_logs)
            loss = model.vae.get_loss(vectorized_logs)
        
            anomaly = False
            if loss > threshold:
                anomaly = True

            output = anomaly
        else:
            raise ValueError("operation mode not recognized")

        after_preprocess_time = time.time()

        compress_and_send(output,operation_mode,i,log_creation_time,log_catch_time,after_preprocess_time)

        #training step
        # metrics["mean_padding"].append(tf.reduce_mean(tf.reduce_sum(tf.cast(vectorized_logs==0,tf.int32),axis=-1)).numpy())
        # metrics["reconstruction_loss"].append(loss)

        #saving model after a number of iterations
        # if i%save_iterations == 0:
        #     # model.vae.save_model(permanent_folder+"/logs_model/")
        #     with open(permanent_folder+"metrics.pkl","wb") as f:
        #         pickle.dump(metrics,f)


            

