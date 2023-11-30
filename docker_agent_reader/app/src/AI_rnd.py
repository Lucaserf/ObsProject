import tensorflow as tf

class AnomalyDetector():
    def __init__(self,latent_space_dim,threshold):
        tf.random.set_seed(42)
        self.threshold = threshold

        inputs = tf.keras.layers.Input(shape=(latent_space_dim,))
        hidden_layer = tf.keras.layers.Dense(128, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(128)(hidden_layer)
        self.fixed_model = tf.keras.Model(inputs, outputs,name="fixed_model")
        self.fixed_model.trainable = False

        inputs = tf.keras.layers.Input(shape=(latent_space_dim,))
        hidden_layer = tf.keras.layers.Dense(128, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(128)(hidden_layer)
        self.trainable_model = tf.keras.Model(inputs, outputs,name="trainable_model")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recostruction_loss")
        self.fixed_model.summary()
        self.trainable_model.summary()

    def train_step(self,data):
        with tf.GradientTape() as tape:
            y_pred = self.trainable_model(data)
            y_true = self.fixed_model(data)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(y_true, y_pred)
                )
            )
        grads = tape.gradient(reconstruction_loss, self.trainable_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_model.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    def detect(self,data,threshold):
        anomaly = False
        y_pred = self.trainable_model(data)
        y_true = self.fixed_model(data)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(y_true, y_pred)
            )
        )
        if reconstruction_loss>threshold:
            anomaly = True
        
        return reconstruction_loss,anomaly

    

