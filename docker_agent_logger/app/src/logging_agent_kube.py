import time
import os
import shutil
import pickle
import pandas as pd
import bz2
from pre_processing import *

root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data"

try:
    os.mkdir(permanent_folder)
except:
    pass


#we give the dataset as a given
df = pd.read_csv("app/src/OpenStack_2k.log_structured.csv")
df = df.drop(["LineId","EventId","EventTemplate"],axis=1)
def time_to_number(time):
    time = time.split(":")
    return float(time[0])*60*60+float(time[1])*60+float(time[2])

df["Pid"] = df["Pid"].apply(str)
df_time = df["Time"].apply(time_to_number)
logs = []
for i,r in df.iterrows():
    logs.append(" ".join(r))

tokenizer = Tokenizer()
tokenizer.training_tokenizer(logs)
i = 0

while True:
    
    data = os.listdir("/var/log/")
    print(data)
    #filtering

    data = [x for x in data if "openstacklogs" in x]

    print(data)

    #log rotation and aggregation

    if len(data)> 32:
        new_logs = []
        new_logs_prep = []
        for d in data:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read()

            new_logs.append(logs)

            #preprocess 
            new_logs_prep.append(tokenizer.preprocess(logs))


            os.remove(data_path)

        with bz2.open(os.path.join(permanent_folder,f"data_{i}.log"),"wb") as f:
            pickle.dump("\n".join(new_logs)+ "\n",f)

        with bz2.open(os.path.join(permanent_folder,f"encoded_data_{i}.log"),"wb") as f:
            pickle.dump(new_logs_prep,f)
    
    time.sleep(10)