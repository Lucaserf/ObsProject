# from cloudevents.http import CloudEvent
# from cloudevents.conversion import to_binary
# import requests
import time
import os
import sys
import tensorflow as tf
from AI import *
import zmq
import time
import msgpack
import pickle
import numpy as np
# import kubernetes as k8s
import socket

print("starting")

root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass

context = zmq.Context()
socket_zmq = context.socket(zmq.PUSH)

operation_mode = os.environ["OPERATION_MODE"]
auto_selection = os.environ["AUTO_SELECTION"]

op_dict = {"logs": 0, "vectorized_logs": 1, "anomaly": 2}
op_dict_inverse = {0: "logs", 1: "vectorized_logs", 2: "anomaly"}

op = op_dict[operation_mode]



def compress_and_send(data,type_log,i,log_creation_time,catching_time):
            
            # compressed_data = bz2.compress(pickle.dumps(data))
            # print(f"lenght of {type_log}: ", sys.getsizeof(compressed_data))
            
            # metrics[type_log].append(sys.getsizeof(compressed_data))

            data = pickle.dumps(data)

            event = {
                "id_node": os.environ["HOSTNAME"].split("-")[-1],
                "id": i,
                "type": type_log,
                "source": "simulation",
                "catch_time": catching_time,
                "after_preprocess_time": time.time(),
                "log_creation_time": log_creation_time,
                "data": data,
            }
            
            compressed_event = pickle.dumps(event)  

            socket_zmq.send(compressed_event)


#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
vocab_size = 5000
max_len=85 # mean length + std length
latent_dim=max_len//2
threshold = 530
number_logs_to_send = os.environ["LOGS_TO_SEND"]
if number_logs_to_send == "inf":
    number_logs_to_send = np.inf
else:
    number_logs_to_send = int(number_logs_to_send)

with open("./app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/15")

i = 0
save_iterations = 20

metrics ={"total_loss":[],"reconstruction_loss":[],"kl_loss":[],"logs":[],"vectorized_logs":[],"encoded_logs":[],"mean_padding":[],"anomaly":[]}

time_last_send = time.time()

endpoints = socket.gethostbyname_ex("reader-service.default.svc.cluster.local")[2]
print(endpoints)
print(len(endpoints))
for ip in endpoints:
    try:
        socket_zmq.connect(f"tcp://{ip}:3000")
        print(f"connected to {ip}")
    except:
        print(f"connection to {ip} failed")

print("connected to all")

# delta_messages = int(number_logs_to_send/3.5)

# delta_messages2 = delta_messages+ int(number_logs_to_send/2)

changed = True
while i < number_logs_to_send:
    
    data = os.listdir(log_folder)
    #filtering

    data = [x for x in data if "BGL" in x]
    data.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder,x)))

    # if auto_selection == "True" and changed:
    #     if len(data) > 1:
    #         op -= 1
    #         changed = False
    #     elif len(data) < 1 and (time.time()-time_last_send)*1000 > 20: #if we wait more than 20 ms we can do more calculations
    #         op += 1
    #         changed = False
    #     else:
    #         pass
    #     if op < 0:
    #         op = 0
    #     if op > 2:
    #         op = 2

    #set times for the selection of the logs 
    # if auto_selection == "True":
    #     if (i+1)%delta_messages == 0 and op == 2:
    #         op-=1
    #     if (i+1)%delta_messages2 == 0 and op == 1:
    #         op-=1
    #     if op < 0:
    #         op = 0
    #     if op > 2:
    #         op = 2


        #i have to select a number of logs to change the operation mode.
    
    

    #log rotation and aggregation
    if len(data)>= 1:
        changed = True
            
        i += 1
        d = data[0]
        
        print(f"doing '{d}'")
        data_path = os.path.join(log_folder,d)

        with open(data_path) as f:
            new_logs = f.read().split("\n")[:-1] #escluding the last line because it is empty

        os.remove(data_path)
    
        log_creation_time = float(d[3:-4])

        log_catch_time = time.time()

        if len(new_logs) > 10 and auto_selection == "True":
            op = 0


        if op == 0:
            output = new_logs
            #filtro

        elif op == 1:
            # output = tokenizer.tokenizer(new_logs).numpy()
            # output = [lv.tolist() for lv in output]
            output = tokenizer.tokenizer(new_logs)

        elif op == 2:  
            # vectorized_logs = tokenizer.vectorization(new_logs)
            # loss = model.vae.get_loss(vectorized_logs)
            # anomaly = []
            # for l in loss:
            #     if l > threshold:
            #         anomaly.append(True)
            #     else:
            #         anomaly.append(False)

            # output = anomaly
            # number_logs = len(new_logs)
            # anomaly_rate = random.random()*0.05
            # anomaly_index_selection = [random.randint(0,number_logs-1) for _ in range(int(number_logs*anomaly_rate))]
            # output = [new_logs[i] for i in anomaly_index_selection]
            output = [log for log in new_logs if random.random() < 0.01]
        else:
            raise ValueError("operation mode not recognized")

        compress_and_send(output,op_dict_inverse[op],i,log_creation_time,log_catch_time)

        time_last_send = time.time()

        #training step
        # metrics["mean_padding"].append(tf.reduce_mean(tf.reduce_sum(tf.cast(vectorized_logs==0,tf.int32),axis=-1)).numpy())
        # metrics["reconstruction_loss"].append(loss)

        #saving model after a number of iterations
        # if i%save_iterations == 0:
        #     # model.vae.save_model(permanent_folder+"/logs_model/")
        #     with open(permanent_folder+"metrics.pkl","wb") as f:
        #         pickle.dump(metrics,f)


            

