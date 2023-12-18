import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import marshal
import time
import collections
import os

from docker_agent_logger.app.src.AI import *
from matplotlib.colors import ListedColormap

vocab_size = 5000
max_len=85
epochs=16
chkpt = "docker_agent_logger/app/classifier_bgl_filtered/15"
MAN_TRAINING_SEQ_LEN = 1000

raw_ds = (
    tf.data.TextLineDataset("persistent_volume/data/BGL/BGL.log")
    # .filter(lambda x: tf.strings.length(x) < MAN_TRAINING_SEQ_LEN)
    # .batch(128)
    # .shuffle(buffer_size=256)
)

# vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
#             raw_ds,
#             vocabulary_size=vocab_size,
#             reserved_tokens=["[PAD]", "[UNK]", "[BOS]","[EOS]"],
#         )

# with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","wb") as f:
#     pickle.dump(vocab,f)

with open("docker_agent_logger/app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)

def get_labels(data: tf.Tensor):
    data = data.decode("utf-8")
    if data[0] == "-":
        return (data[2:],False)
    else:
        return (data,True)
    



ds = raw_ds.map(lambda x: tf.numpy_function(func=get_labels,inp=[x],Tout=(tf.string,tf.bool)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# ds_tokenized = ds.map(lambda x,y: (tf.numpy_function(func=tokenizer.vectorization,inp=[x],Tout=tf.int32),y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
#     tf.data.AUTOTUNE
# )
ds_tokenized = ds.map(lambda x,y: (tokenizer.vectorization(x)[0],y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)



val_split = 0.2
ds_size = 4747963

train_size = int((1-val_split) * ds_size)
val_size = int(val_split * ds_size)

train_size = 3496193

train_ds = ds.take(train_size).filter(lambda x,y: y == False)
train_ds = train_ds.map(lambda x,y: tokenizer.vectorization(x)[0], num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.shuffle(buffer_size=train_size).batch(128)
val_ds = ds.skip(train_size).take(val_size)



model = Model(vocab_size = vocab_size,latent_dim=max_len//2,embedding_dim=128,max_len = max_len)

model.vae.load_model(chkpt=chkpt) #17 for the other model

# model.train_model(ds,epochs=epochs,chkpt=chkpt)




percentile = 98
# anomaly_detector = AnomalyDetector(latent_space_dim=max_len//2,threshold=np.inf)
data = {"logs":[],"parsed_logs":[],"vectorized_logs":[],"encoded_logs":[]}
times = {"parsed_logs":[],"vectorized_logs":[],"encoded_logs":[],"anomaly":[],"anomaly_rnd":[]}
recostruction_loss = []
# recostruction_loss_rnd = []
d = collections.deque(maxlen=1000)
# d_rnd = collections.deque(maxlen=1000)
# thresholds = [np.inf,]
thresholds = [530,]
# thresholds_rnd = [np.inf,]


FP_rec = []
TP_rec = []
FN_rec = []
TN_rec = []

# FP_rnd = []
# TP_rnd = []
# FN_rnd = []
# TN_rnd = []


raw_ds = (
    tf.data.TextLineDataset("persistent_volume/data/BGL/test_set_bgl.txt")
    # .filter(lambda x: tf.strings.length(x) < MAN_TRAINING_SEQ_LEN)
    # .batch(128)
    # .shuffle(buffer_size=256)
)
ds = raw_ds.map(lambda x: tf.numpy_function(func=get_labels,inp=[x],Tout=(tf.string,tf.bool)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)



for logs in ds.batch(1):
    # t = time.time()
    # parsed_logs = tokenizer.parsing(logs)   
    t_parse = time.time()
    # times["parsed_logs"].append(t_parse-t)
    vectorized_logs = tokenizer.vectorization(logs[0])
    t_vectorize = time.time()
    times["vectorized_logs"].append(t_vectorize-t_parse)
    encoded_logs = model.vae.encode(vectorized_logs)
    t_encode = time.time()
    times["encoded_logs"].append(t_encode-t_vectorize)
    loss = model.vae.get_loss(vectorized_logs)

    anomaly = logs[1].numpy()[0]
    if loss > thresholds[-1]:
        if anomaly:
            TP_rec.append(logs[0].numpy()[0].decode("utf-8"))
        else:
            FP_rec.append(logs[0].numpy()[0].decode("utf-8"))
    else:
        if anomaly:
            FN_rec.append(logs[0].numpy()[0].decode("utf-8"))
        else:
            TN_rec.append(logs[0].numpy()[0].decode("utf-8"))

    # d.append(loss)
    # thresholds.append(np.percentile(d,percentile))
    
    t_anomaly = time.time()
    times["anomaly"].append(t_anomaly-t_encode)

    # recostruction_loss_rnd_value, anomaly_rnd = anomaly_detector.detect(encoded_logs,thresholds_rnd[-1])

    # if anomaly_rnd:
    #     if anomaly:
    #         TP_rnd.append(logs[0].numpy()[0].decode("utf-8"))
    #     else:
    #         FP_rnd.append(logs[0].numpy()[0].decode("utf-8"))
    # else:
    #     if anomaly:
    #         FN_rnd.append(logs[0].numpy()[0].decode("utf-8"))
    #     else:
    #         TN_rnd.append(logs[0].numpy()[0].decode("utf-8"))

    # d_rnd.append(recostruction_loss_rnd_value.numpy())
    # thresholds_rnd.append(np.percentile(d_rnd,percentile))
    

    # times["anomaly_rnd"].append(time.time()-t_anomaly)

    # anomaly_detector.train_step(encoded_logs)


    # recostruction_loss_rnd.append(recostruction_loss_rnd_value.numpy())
    recostruction_loss.append(loss)
    compressed_data = bz2.compress(pickle.dumps(logs))
    data["logs"].append(sys.getsizeof(compressed_data))
    # compressed_data = bz2.compress(pickle.dumps(parsed_logs))
    # data["parsed_logs"].append(sys.getsizeof(compressed_data))
    compressed_data = bz2.compress(pickle.dumps(vectorized_logs))
    data["vectorized_logs"].append(sys.getsizeof(compressed_data))
    compressed_data = bz2.compress(pickle.dumps(encoded_logs))
    data["encoded_logs"].append(sys.getsizeof(compressed_data))


    

fig, ax = plt.subplots()
for key in data.keys():
    ax.plot(data[key], label=key)
    plt.xlabel('logs')
    plt.ylabel('size [Bytes]')
    plt.legend()

fig.savefig("data/size.png")

fig, ax = plt.subplots()
for key in times.keys():
    ax.plot(np.array(times[key][10:])*10**3, label=key)
    plt.xlabel('logs')
    plt.ylabel('time [ms]')
    plt.legend()

fig.savefig("data/time.png")

c = np.array(list((ds.batch(1).map(lambda x,y: int(y[0])).take(len(recostruction_loss)).as_numpy_iterator())))

fig, ax = plt.subplots()
# ax.plot(thresholds)
cmap = ListedColormap(["blue","red"])
ax.scatter([i for i in range(len(recostruction_loss))],recostruction_loss, c = c,cmap = cmap,s=0.1)
# ax.scatter(recostruction_loss)
plt.xlabel('logs')
plt.ylabel('reconstruction loss')

fig.savefig("data/reconstruction_loss.png")



with open("data/anomalies.txt","w") as f:
    f.write("true anomalies: {}\n".format(len(TP_rec)+len(FN_rec)))
    f.write("false labeled anomalies: {}\n".format(len(FP_rec)))
    f.write("true labeled anomalies: {}\n".format(len(TP_rec)))
    f.write("precision: {}\n".format(len(TP_rec)/(len(TP_rec)+len(FP_rec))))
    f.write("accuracy: {}\n".format((len(TP_rec)+len(TN_rec))/(len(TP_rec)+len(TN_rec)+len(FP_rec)+len(FN_rec))))
    f.write("F1-score: {}\n".format(2*len(TP_rec)/(2*len(TP_rec)+len(FP_rec)+len(FN_rec))))
    # f.write("false labeled anomalies rnd: {}\n".format(len(FP_rnd)))
    # f.write("true labeled anomalies rnd: {}\n".format(len(TP_rnd)))
    # f.write("precision rnd: {}\n".format(len(TP_rnd)/(len(TP_rnd)+len(FP_rnd))))
    # f.write("accuracy rnd: {}\n".format((len(TP_rnd)+len(TN_rnd))/(len(TP_rnd)+len(TN_rnd)+len(FP_rnd)+len(FN_rnd))))
    # f.write("F1-score rnd: {}\n".format(2*len(TP_rnd)/(2*len(TP_rnd)+len(FP_rnd)+len(FN_rnd))))

    f.write("true anomalies:\n")
    for i in TP_rec+FN_rec:
        f.write(i+"\n")
    f.write("false labeled anomalies:\n")
    for i in FP_rec:
        f.write(i+"\n")
    f.write("true labeled anomalies:\n")
    for i in TP_rec:
        f.write(i+"\n")