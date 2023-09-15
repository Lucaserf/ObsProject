import time
import os
import shutil
import pickle
from pre_processing import preprocess

root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data"

try:
    os.mkdir(permanent_folder)
except:
    pass


while True:
    
    data = os.listdir("/var/log/")
    print(data)
    #filtering

    data = [x for x in data if "quotes" in x]

    print(data)

    #log rotation and aggregation

    if len(data)> 0:
        new_logs = []
        new_logs_prep = []
        for d in data:
            data_path = os.path.join(log_folder,d)

            with open(data_path) as f:
                logs = f.read()

            new_logs.append(logs)

            
        
            #preprocess 
            new_logs_prep.append(preprocess(logs))


            os.remove(data_path)

        with open(os.path.join(permanent_folder,"data.log"),"a") as f:
            f.write("\n".join(new_logs)+ "\n")

        with open(os.path.join(permanent_folder,"encoded_data.log"),"ab") as f:
            pickle.dump(tokens,f)
    
    time.sleep(10)