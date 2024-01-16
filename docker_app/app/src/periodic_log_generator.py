import time
import tensorflow as tf
import os


data_folder = "/var/data/"
permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass

raw_ds = (
    tf.data.TextLineDataset(data_folder+"BGL.log")
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

log_generation_time = []


for i,log in enumerate(val_ds):
    t = time.time()
    with open("/var/log/BGL{}.log".format(i),"w") as f:
        f.write(log[0].numpy().decode("utf-8")+"\n")
    # with open(permanent_folder+"log_generation_time.txt","a") as f:
    #     f.write("{}\n".format(str(time.time())))
    t_el = (time.time()-t)*1000
    print("generated log {}, after {} ms".format(i,t_el))
    time.sleep(10e-3)


print("the dataset is finished")
