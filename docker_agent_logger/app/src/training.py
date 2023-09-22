import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2
import transformers
from AI import *
import tensorflow as tf
import random


df = pd.read_csv("docker_agent_logger/app/data/OpenStack_2k.log_structured.csv")

labels = tf.constant(df["EventId"].apply(lambda x: int(x[1:])-1))

df = df.drop(["LineId","EventId","EventTemplate"],axis=1)

df["Pid"] = df["Pid"].apply(str)

logs = []

for i,r in df.iterrows():
    logs.append(" ".join(r))

vocab_size = 32000
max_len=512
labels_size=len(np.unique(labels))
epochs=200
chkpt = "docker_agent_logger/app/classifier/"

tokenizer = Tokenizer("./docker_agent_logger/app/logs_tokenizer",vocab_size=vocab_size,max_len=max_len)

tokenized_data = tokenizer.preprocess(logs)["input_ids"]
val_split = 0.2
ds_size = len(labels)

dataset = tf.data.Dataset.from_tensor_slices((tokenized_data,labels))

# dataset = dataset.shuffle(dataset.cardinality(),seed=42)

train_size = int((1-val_split) * ds_size)
val_size = int(val_split * ds_size)

train_ds = dataset.take(train_size).shuffle(train_size,seed=42).batch(16)
val_ds = dataset.skip(train_size).take(val_size).batch(16)


model = Model(vocab_size = vocab_size,labels_size = labels_size,max_len = max_len)

model.train_model(train_ds,val_ds,epochs=epochs,chkpt=chkpt)







