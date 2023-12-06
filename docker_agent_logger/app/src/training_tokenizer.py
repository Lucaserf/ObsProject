import sys
sys.dont_write_bytecode = True
from AI import *
import keras_nlp
import pickle
import tensorflow as tf


vocab_size = 5000

raw_ds = ( #.filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    tf.data.TextLineDataset("persistent_volume/data/BGL/BGL.log")
    .batch(128)
    .shuffle(buffer_size=256)
)

# def parsing(data):
#     data = tf.strings.regex_replace(data, r'\b[a-zA-Z\d\-_\.]{20,}\b', '*')
#     return data


# ds = raw_ds.map(parsing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
#     tf.data.AUTOTUNE
# )

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            raw_ds,
            vocabulary_size=vocab_size,
            reserved_tokens=["[PAD]", "[UNK]","[BOS]","[EOS]"],
        )

with open("docker_agent_logger/app/logs_tokenizer/vocab_bgl.pkl","wb") as f:
    pickle.dump(vocab,f)