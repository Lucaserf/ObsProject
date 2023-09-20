import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2
import transformers
from pre_processing import Tokenizer



df = pd.read_csv("docker_agent_logger/app/src/OpenStack_2k.log_structured.csv")

labels = df["EventId"]

df = df.drop(["LineId","EventId","EventTemplate"],axis=1)

df["Pid"] = df["Pid"].apply(str)

logs = []

for i,r in df.iterrows():
    logs.append(" ".join(r))

tokenizer = Tokenizer("./docker_agent_logger/app/bert-base-cased_saved")

reduced_logs = logs
encoded_data = tokenizer.preprocess(reduced_logs)



