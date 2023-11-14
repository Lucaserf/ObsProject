import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2
from AI import *
import tensorflow as tf
import random
import matplotlib.pyplot as plt


# df = pd.read_csv("docker_agent_logger/app/data/OpenStack_2k.log_structured.csv")

# labels = tf.constant(df["EventId"].apply(lambda x: int(x[1:])-1))

# df = df.drop(["LineId","EventId","EventTemplate"],axis=1)

# df["Pid"] = df["Pid"].apply(str)

# logs = []

# for i,r in df.iterrows():
#     logs.append(" ".join(r))



vocab_size = 4000
max_len=60
epochs=64
MAX_TRAINING_SEQ_LEN = 3000
chkpt = "docker_agent_logger/app/classifier/"

raw_ds = ( #
    tf.data.TextLineDataset("persistent_volume/data/HDFS_v2/node_logs/hadoop-hdfs-datanode-mesos-01.log")
    .filter(lambda x: tf.strings.length(x) < MAX_TRAINING_SEQ_LEN)
    .batch(128)
    .shuffle(buffer_size=516)
)

# vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
#             raw_ds,
#             vocabulary_size=vocab_size,
#             reserved_tokens=["[PAD]", "[UNK]","[SEP]","[BOS]","[EOS]"],
#         )

# with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","wb") as f:
#     pickle.dump(vocab,f)

with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)


ds = raw_ds.map(tokenizer.preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

#get max length of sequences in ds
# stats = OnlineStats()
# for i in ds:
#     stats.update(len(i.to_list()[0]))

# print("max len:",stats.get_max())
# print("mean len:",stats.get_mean())
# print("std len:",stats.get_std())

# val_split = 0.2
# ds_size = ds.cardinality().numpy()

# train_size = int((1-val_split) * ds_size)
# val_size = int(val_split * ds_size)

# train_ds = ds.take(train_size)
# val_ds = ds.skip(train_size).take(val_size)



model = Model(vocab_size = vocab_size,latent_dim=max_len//2,embedding_dim=128,max_len = max_len)

model.vae.load_model(chkpt=chkpt+"test35")
 
model.train_model(ds,epochs=epochs,chkpt=chkpt+"test35")



# def plot_label_clusters(vae, data):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = vae.encoder.predict(data)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1])
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig("cluster.png")

# plot_label_clusters(model.vae, ds)

# z = tf.random.normal(shape=(1, 256))
# encode_token = ds.take(1).as_numpy_iterator().next()

# print(tokenizer.decode(encode_token))

# tokens = model.vae.decode(z)

# print(tokenizer.decode(tokens))


