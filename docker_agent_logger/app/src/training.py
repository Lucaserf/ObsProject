import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2
import transformers
from AI import *
import tensorflow as tf
import random


# df = pd.read_csv("docker_agent_logger/app/data/OpenStack_2k.log_structured.csv")

# labels = tf.constant(df["EventId"].apply(lambda x: int(x[1:])-1))

# df = df.drop(["LineId","EventId","EventTemplate"],axis=1)

# df["Pid"] = df["Pid"].apply(str)

# logs = []

# for i,r in df.iterrows():
#     logs.append(" ".join(r))



vocab_size = 5000
max_len=512
epochs=200
chkpt = "docker_agent_logger/app/classifier/"

raw_ds = ( #.filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    tf.data.TextLineDataset("docker_agent_logger/app/data/HDFS_v2/node_logs/hadoop-hdfs-datanode-mesos-32.log")
    .batch(64)
    .shuffle(buffer_size=256)
)

# vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
#             raw_ds,
#             vocabulary_size=vocab_size,
#             reserved_tokens=["[PAD]", "[UNK]", "[BOS]","[EOS]"],
#         )

# with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","wb") as f:
#     pickle.dump(vocab,f)

with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)


ds = raw_ds.map(tokenizer.preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

val_split = 0.2
ds_size = ds.cardinality().numpy()

train_size = int((1-val_split) * ds_size)
val_size = int(val_split * ds_size)


train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)


model = Model(vocab_size = vocab_size,latent_dim=256,embedding_dim=128,max_len = max_len)

model.train_model(train_ds,val_ds,epochs=epochs,chkpt=chkpt)







