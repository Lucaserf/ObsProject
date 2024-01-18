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
    tf.data.TextLineDataset(data_folder+"test_set_bgl.txt")
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

# val_split = 0.2
# ds_size = 4747963

# train_size = int((1-val_split) * ds_size)
# val_size = int(val_split * ds_size)


# val_ds = ds.skip(train_size).take(val_size)


t = time.time()
for i,log in enumerate(ds):
    
    time.sleep(100e-3)
    with open("/var/log/BGL{}.log".format(i),"w") as f:
        f.write(log[0].numpy().decode("utf-8")+"\n")

    print("generated log {}, after {} ms".format(i,(time.time()-t)*1000))
    t = time.time()
    


print("the dataset is finished")
