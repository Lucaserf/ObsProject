

import transformers
import tensorflow as tf
from keras_nlp.layers import TokenAndPositionEmbedding,TransformerEncoder
import numpy as np

class Tokenizer():
    def __init__(self,path,vocab_size = 32000, max_len=512):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path,max_lenth = max_len)
        # tokens = tokenizer(sentences,return_tensors="np",padding="max_length",max_length=50,return_attention_mask=False,return_token_type_ids =False)["input_ids"]

    def training_tokenizer(self,training_corpus):
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)

    def saving_tokenizer(self,path):
        self.tokenizer.save_pretrained(path)

    def preprocess(self,data):
        # tokens = self.tokenizer(data)["input_ids"]
        tokens = self.tokenizer(data,padding="max_length",truncation=True,return_tensors="tf")
        return tokens


    def decode(self,data):
        decoded_data = self.tokenizer.batch_decode(data)
        return decoded_data

class Model():
    def __init__(self,vocab_size,labels_size,max_len):

        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')

        input_embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=max_len,
            embedding_dim=128,
            mask_zero=True,
            name = "input_embedding"
        )(input_ids)

        encoding = TransformerEncoder(
            num_heads=4,
            intermediate_dim=256,
            name = "encoder"
        )(input_embedding)

        labels = tf.keras.layers.Dense(labels_size, name="output")(encoding[:,0,:])

        self.model = tf.keras.Model(inputs=[input_ids],
            outputs=[labels],
            name = "classifier")
        
        self.opt_model = tf.keras.optimizers.Adam(learning_rate=1e-5)

        self.model.summary()

    def train_model(self,train_ds,val_ds,epochs,chkpt):
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self.opt_model,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            jit_compile=False
        )

        es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks = [es_cb,cp_cb]
        )


        
        





    

