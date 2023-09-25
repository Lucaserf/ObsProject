import time
import os
import shutil
import pickle
import pandas as pd
import bz2
# from pre_processing import *



permanent_folder = "./var/log/pv/logging_data/"
# tokenizer = Tokenizer("./app/logs_tokenizer")

while True:

    data = os.listdir(permanent_folder)

    data_encoded = [x for x in data if "encoded_data_" in x]
    data_encoded.sort(key=lambda x: os.path.getmtime(os.path.join(permanent_folder,x)))

    for data_path in data_encoded:
        data_path = os.path.join(permanent_folder,data_path)

        with bz2.open(data_path,"rb") as f:
            new_logs_prep = pickle.load(f)

        os.remove(data_path)
        print(new_logs_prep[:2])

    
    for data_path in data:
        try:
            os.remove(data_path)
        except:
            pass

    time.sleep(2)