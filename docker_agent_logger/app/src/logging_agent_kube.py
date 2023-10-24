import time
import os
import shutil
import pickle
import pandas as pd
import bz2
from pre_processing import *

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

    if len(data)> 32:

        new_logs = []
        new_logs_prep = []
        for d in data:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read().split("\n")[:-1]

            new_logs += logs

            #preprocess 
            new_logs_prep+= tokenizer.preprocess(logs)


            os.remove(data_path)

        with open(os.path.join(permanent_folder,f"raw_data_{i}.log"),"a") as f:
            f.write("\n".join(new_logs))

        with bz2.open(os.path.join(permanent_folder,f"data_{i}.log"),"wb") as f:
            pickle.dump("\n".join(new_logs),f)

        with bz2.open(os.path.join(permanent_folder,f"encoded_data_{i}.log"),"wb") as f:
            pickle.dump(new_logs_prep,f)
        i+= 1
    time.sleep(10)