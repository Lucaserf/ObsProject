import zmq
import pickle
import os
import random
import sys
import time

context = zmq.Context()

socket = context.socket(zmq.PUSH)
socket.connect("tcp://192.168.17.36:3000")

dimension_pack = int(os.environ["DIMENSION_PACK"])
period = float(os.environ["period"])

pack = str([random.random() for i in range(dimension_pack)])
dim = sys.getsizeof(pickle.dumps(pack))
print(dim)
print("starting sending data")
while True:
    t = time.time()
    socket.send(pickle.dumps((str(t),dim,pack)))
    time.sleep(period-(time.time()-t))
