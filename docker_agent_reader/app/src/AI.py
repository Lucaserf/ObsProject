import tensorflow as tf

class AnomalyDetector():
    def __init__(self,latent_space_dim,threshold):
        tf.random.set_seed(42)
        self.threshold = threshold
        self.fixed_model = tf.keras.Sequential(
                [   
                    tf.keras.layers.InputLayer(input_shape=(latent_space_dim,), name="input"),
                    tf.keras.layers.Dense(128, activation="relu", name="layer1"),
                    tf.keras.layers.Dense(128, activation="relu", name="layer2"),
                    tf.keras.layers.Dense(128, name="layer3"),
                ]
            )
        self.fixed_model.trainable = False
        self.trainable_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_space_dim,), name="input"),
                tf.keras.layers.Dense(128, activation="relu", name="layer1"),
                tf.keras.layers.Dense(128, activation="relu", name="layer2"),
                tf.keras.layers.Dense(128, name="layer3"),
            ]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recostruction_loss")
        self.fixed_model.summary()

    def train_step(self,data):
        with tf.GradientTape() as tape:
            y_pred = self.trainable_model(data)
            y_true = self.fixed_model(data)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(y_true, y_pred), axis=(1)
                )
            )
        grads = tape.gradient(reconstruction_loss, self.trainable_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_model.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    def detect(self,data):
        anomaly = False
        y_pred = self.trainable_model(data)
        y_true = self.fixed_model(data)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(y_true, y_pred), axis=(1)
            )
        )
        if reconstruction_loss>self.threshold:
            anomaly = True
        
        return reconstruction_loss,anomaly

    

