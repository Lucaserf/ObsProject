import zmq
import pickle
import time
import os

context = zmq.Context()

permanent_folder = "var/log/pv/logging_data/"

try:
    os.mkdir(permanent_folder)
except:
    pass

with open(permanent_folder+"data_time.txt","w") as f:
    f.write("catch_time\n")

socket = context.socket(zmq.PULL)
socket.bind("tcp://*:3000")
print("start waiting for data")


while True:
    message = socket.recv()
    t = time.time()
    t, dim, _ = pickle.loads(message)
    with open(permanent_folder+"data_time.txt","a") as f:
        f.write(f"{dim},{time.time()-t}\n")