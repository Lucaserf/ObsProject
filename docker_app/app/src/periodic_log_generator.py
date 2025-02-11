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
    tf.data.TextLineDataset(data_folder+os.environ["DATANAME"])
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
starting_gen_period = gen_period

batch, change = os.environ["BATCH_SIZE"].split(",")
batch = int(batch)


fmax = 1/gen_period_min
fmin = 1/gen_period_max

#linear frequency change every 10 seconds
period_change = 20

deltaf = (fmax-fmin)/6

t_change = time.time()+period_change
t = time.time()

batch_count = 0

t_start = time.time()

for i,log in enumerate(ds.batch(1)):

    #change batch size at a specific frequency, check every 15 seconds, to scale with batch before the scaling changing pre-processing
    if (time.time() - t_start) > 82 and batch != 16 and change=="True":
        batch = 16
        print("batch size changed to 32")

    if period_change-(time.time()-t_change) < 0:
        #linear change
        gen_period = 1/((1/gen_period)+deltaf)
        if gen_period < gen_period_min:
            gen_period = gen_period_min
        t_change = time.time()

    time_to_wait = gen_period-(time.time()-t)
    if time_to_wait > 0:
        time.sleep(time_to_wait)

    # with open("/var/log/tmp.log","a") as f:
    #     f.write("\n".join([x.decode("utf-8") for x in log[0].numpy()])+"\n")
    with open("/var/log/tmp.log","a") as f:
        f.write(log[0].numpy()[0].decode("utf-8")+"\n")
        
    batch_count += 1
    if batch_count == batch:
        batch_count = 0
        os.rename("/var/log/tmp.log","/var/log/BGL{}.log".format(str(time.time())))
    
    # print("generated log {}, after {} ms".format(i,(time.time()-t)*1000))
    
    t = time.time()
    

print("the dataset is finished")
