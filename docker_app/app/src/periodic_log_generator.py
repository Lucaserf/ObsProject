import time
import tensorflow as tf
import os
import numpy as np



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

np.random.seed(int(os.environ["SEED"]))

# val_split = 0.2
# ds_size = 4747963

# train_size = int((1-val_split) * ds_size)
# val_size = int(val_split * ds_size)


# val_ds = ds.skip(train_size).take(val_size)

start_time = float(os.environ["START_TIME"])

#sleep until 100 seconds after start_time
sync_time = float(os.environ["WAIT_TIME"])
desync_time = sync_time-(time.time()-start_time)
if  desync_time > 0:
    time.sleep(desync_time)

gen_period = os.environ["GEN_PERIOD"].split(",")
gen_period_min, gen_period_max = float(gen_period[0]), float(gen_period[1])
gen_period = gen_period_max

batch = int(os.environ["BATCH_SIZE"])

period_change = 1 #seconds
t_change = time.time()+period_change+1
t = time.time()

for i,log in enumerate(ds.batch(batch)):
    if period_change-(time.time()-t_change) < 0:
        new_speed = np.random.normal(0.5,0.2)
        while new_speed < 0 or new_speed > 1:
            new_speed = np.random.normal(0.5,0.2)
        gen_period = (gen_period_max-gen_period_min)*new_speed+gen_period_min
        t_change = time.time()
    time_to_wait = gen_period-(time.time()-t)
    if time_to_wait > 0:
        time.sleep(time_to_wait)
    
    with open("/var/log/tmp.log","w") as f:
            f.write("\n".join([x.decode("utf-8") for x in log[0].numpy()])+"\n")

    os.rename("/var/log/tmp.log","/var/log/BGL{}.log".format(str(time.time())))
    

    print("generated log {}, after {} ms".format(i,(time.time()-t)*1000))
    
    t = time.time()
    

print("the dataset is finished")
