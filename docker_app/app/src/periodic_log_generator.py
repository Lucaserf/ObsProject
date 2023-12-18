import time
import tensorflow as tf


data_folder = "/var/data/"

raw_ds = (
    tf.data.TextLineDataset("persistent_volume/data/BGL/BGL.log")
    # .filter(lambda x: tf.strings.length(x) < MAN_TRAINING_SEQ_LEN)
    # .batch(128)
    # .shuffle(buffer_size=256)
)

def get_labels(data: tf.Tensor):
    data = data.decode("utf-8")
    if data[0] == "-":
        return (data[2:],False)
    else:
        return (data,True)

ds = raw_ds.map(lambda x: tf.numpy_function(func=get_labels,inp=[x],Tout=(tf.string,tf.bool)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

val_split = 0.2
ds_size = 4747963

train_size = int((1-val_split) * ds_size)
val_size = int(val_split * ds_size)


val_ds = ds.skip(train_size).take(val_size)

for i,log in enumerate(val_ds):
    with open("/var/log/BGL{}.log".format(i),"w") as f:
        f.write(log[0].numpy().decode("utf-8")+"\n")
    time.sleep(0.1)

print("the dataset is finished")

while True:
    time.sleep(10)