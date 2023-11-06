import time
import pandas as pd 



data_folder = "/var/data/"
#first datanode hdfsv2
with open(data_folder+"HDFS_v2/node_logs/hadoop-hdfs-datanode-mesos-01.log") as f:
    logs = f.read().split("\n")[:-1]

done = False
START = time.time()
counter_prints = 0
max_len = len(logs)

while not done:

    with open("/var/log/HDFS{}.log".format(counter_prints),"w") as f:
            f.write(logs[counter_prints]+"\n")
    counter_prints +=1
    time.sleep(0.05)

    if counter_prints == max_len:
        done = True

print("the dataset is finished")