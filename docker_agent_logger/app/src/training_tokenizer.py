import sys
sys.dont_write_bytecode = True
from AI import *
import keras_nlp
import pickle
import tensorflow as tf


vocab_size = 4000
max_len=256
chkpt = "docker_agent_logger/app/classifier/"

raw_ds = ( #.filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    tf.data.TextLineDataset("docker_agent_logger/app/data/HDFS_v2/node_logs/hadoop-hdfs-datanode-mesos-01.log")
    .batch(32)
    .shuffle(buffer_size=256)
)

def parsing(data):
    data = tf.strings.regex_replace(data, r'\b[a-zA-Z\d\-_\.]{20,}\b', '*')
    return data


ds = raw_ds.map(parsing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            ds,
            vocabulary_size=vocab_size,
            reserved_tokens=["[PAD]", "[UNK]","[SEP]","[BOS]","[EOS]"],
        )

with open("docker_agent_logger/app/logs_tokenizer/vocab.pkl","wb") as f:
    pickle.dump(vocab,f)