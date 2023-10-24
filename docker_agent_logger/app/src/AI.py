

import tensorflow as tf
import keras_nlp
import numpy as np
import re
import random

class Tokenizer():
    def __init__(self,vocab, max_len=512):
        
        self.tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=vocab,
            sequence_length=max_len
        )

        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=max_len,
            start_value=self.tokenizer.token_to_id("[BOS]"),
            end_value= self.tokenizer.token_to_id("[EOS]"),
        )

    def preprocess(self,data):
        data = tf.strings.regex_replace(data, r'\b[a-zA-Z\d\-_\.]{20,}\b', '*')
        tokens = self.tokenizer(data)
        features = self.start_packer(tokens)
        labels = tokens
        return tokens


    def decode(self,data):
        decoded_data = self.tokenizer.detokenize(data)
        return decoded_data

class Model():
    def __init__(self,vocab_size,latent_dim,embedding_dim,max_len):

        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')

        input_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=max_len,
            embedding_dim=embedding_dim,
            mask_zero=True,
            name = "input_embedding"
        )(input_ids)

        encoding = keras_nlp.layers.TransformerEncoder(
            num_heads=4,
            intermediate_dim=latent_dim,
            name = "encoder"
        )(input_embedding)

        z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(encoding[:,0,:])
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(encoding[:,0,:])

        z = Sampling(name="z")([z_mean, z_log_var])

        self.encoder = tf.keras.Model(inputs=[input_ids],
            outputs=[z_mean, z_log_var,z],
            name = "classifier")

        self.encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='decoder_input')
        latent_space_input = tf.keras.layers.Input(shape=(max_len,latent_dim,), dtype=tf.float32, name='latent_space_input')
        decoder_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=max_len,
            embedding_dim=latent_dim,
            mask_zero=True,
            name = "decoder_embedding"
        )(decoder_input)

        decoding = keras_nlp.layers.TransformerDecoder(
            num_heads=4,
            intermediate_dim=latent_dim,
            name = "decoder"
        )(decoder_embedding,latent_space_input)

        output = tf.keras.layers.Dense(vocab_size, name="output")(decoding)

        self.decoder = tf.keras.Model(inputs=[decoder_input,latent_space_input],
            outputs=[output],
            name = "decoder")

        self.decoder.summary()

        self.vae = VAE(encoder=self.encoder,decoder=self.decoder)

        

    def train_model(self,train_ds,epochs,chkpt):


        self.vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))


        # es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        # cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath = chkpt, monitor='total_loss', verbose=0, save_best_only=True, mode='auto')
        cp_cb = SavingCallback(chkpt)

        self.vae.fit(
            train_ds,
            epochs=epochs,
            callbacks = [cp_cb]
        )

class SavingCallback(tf.keras.callbacks.Callback):
    def __init__(self,chkpt):
        self.chkpt = chkpt
    def on_epoch_end(self,epoch,logs=None):
        self.model.save_model(self.chkpt)
    

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VAE(tf.keras.Model):
    def __init__(self,encoder,decoder,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recostruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var,z = self.encoder(data)
            z = tf.expand_dims(z,axis=1)*tf.ones((1,512,256))
            reconstruction = self.decoder([data,z])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction, from_logits=True), axis=(1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self,data):
        z_mean, z_log_var,z = self.encoder(data)
        z = tf.expand_dims(z,axis=1)*tf.ones((1,512,256))
        reconstruction = self.decoder([data,z])
        return reconstruction
    
    def save_model(self,chkpt):
        self.encoder.save_weights(chkpt+"encoder")
        self.decoder.save_weights(chkpt+"decoder")

    def load_model(self,chkpt):
        self.encoder.load_weights(chkpt+"encoder")
        self.decoder.load_weights(chkpt+"decoder")

    def encode(self,data):
        z_mean, z_log_var,z = self.encoder(data)
        return z
    def decode(self,data,z):
        z = tf.expand_dims(z,axis=1)*tf.ones((1,512,256))
        reconstruction = self.decoder([data,z])
        return reconstruction

    # def test_step(self,data):
    #     z_mean, z_log_var,z = self.encoder(data)
    #     z = tf.expand_dims(z,axis=1)*tf.ones((1,512,256))
    #     reconstruction = self.decoder([data,z])
    #     reconstruction_loss = tf.reduce_mean(
    #         tf.reduce_sum(
    #             tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction, from_logits=True), axis=(1)
    #         )
    #     )
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #     total_loss = reconstruction_loss + kl_loss
    #     self.total_loss_tracker.update_state(total_loss)
    #     return {
    #         "total_loss": self.total_loss_tracker.result()
    #     }