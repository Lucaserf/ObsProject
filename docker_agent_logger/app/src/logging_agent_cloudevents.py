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
from AI import *


root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass


#we give the dataset as a given to train the tokenizer, for a real application we would have a fase of training and then inference
tokenizer = Tokenizer("./app/logs_tokenizer")
i = 0

while True:
    
    data = os.listdir(log_folder)
    #filtering

    data = [x for x in data if "openstacklogs" in x]
    data.sort(key=lambda x: os.path.getmtime(os.path.join(log_folder,x)))

    #log rotation and aggregation
    if len(data)> 1024:

        new_logs = []
        new_logs_prep = []
        for d in data:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read().split("\n")[:-1]

            new_logs += logs

            #preprocess 
            new_logs_prep += tokenizer.preprocess(logs)

            os.remove(data_path)

        attributes = {
            "type": "logs",
            "source": "simulation",
        }
        
        
        event = CloudEvent(attributes, {"data": new_logs_prep})

        # Creates the HTTP request representation of the CloudEvent in binary content mode
        headers, body = to_binary(event)
        # t = time.time()
        # print("sending encoded data:")

        # r = requests.post("http://10.99.147.168:3000", data=body, headers=headers)

        # print(f"time to send: {time.time()-t}")

        # event = CloudEvent(attributes, {"data": new_logs})

        # headers, body = to_binary(event)
        # t = time.time()
        # print("sending raw data:")

        # r = requests.post("http://10.99.147.168:3000", data=body, headers=headers)

        # print(f"time to send: {time.time()-t}")

        compressed_data = bz2.compress(pickle.dumps(new_logs_prep))
        print("lenght of encoded compressed data: ",str(sys.getsizeof(compressed_data)))

        print("sending compressed encoded data:")
        times = []
        for _ in range(100):
            t=time.time()
            r = requests.post("http://10.99.147.168:3000",data=compressed_data,headers=headers)
            times.append(time.time()-t)
        print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")

        compressed_data = bz2.compress(pickle.dumps(new_logs))
        print("lenght of compressed data: ",sys.getsizeof(compressed_data))

        print("sending compressed data:")
        times = []
        for _ in range(100):
            t=time.time()
            r = requests.post("http://10.99.147.168:3000",data=compressed_data,headers=headers)
            times.append(time.time()-t)
        print(f"time to send: {np.mean(times)} +- {np.std(times)}\n")

    time.sleep(2)

        

