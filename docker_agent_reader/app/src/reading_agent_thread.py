
import threading
import pickle
import sys
import os
from AI import *
import time
import zmq
import time
import msgpack


def process_event(event,server_catch_time,data_size):

    data = event["data"]
    id_node = event["id_node"]
    id = event["id"]
    type_log = event["type"]
    catch_time = event["catch_time"]
    log_creation_time = event["log_creation_time"]
    time_after_preprocess = event["after_preprocess_time"]

    if type_log == "anomaly":
        for i,l in enumerate(data):
            if l:
                print(f"anomaly detected in {i} for event {id} in node {id_node}")

    elif type_log == "logs":
        # parsed_logs = tokenizer.parsing(data)
        vectorized_logs = tokenizer.vectorization(data)
        loss = model.vae.get_loss(vectorized_logs)
        for i,l in enumerate(loss):
            if l > threshold:
                print(f"anomaly detected in {i} for event {id} in node {id_node} with a reconstruction loss of {l}")
                break 
    # elif event["type"] == "parsed_logs":
    #     vectorized_logs = tokenizer.vectorization(data)
    #     loss = model.vae.get_loss(vectorized_logs)
    #     if loss > threshold:
    #         print(f"anomaly detected in {event['id']} with a reconstruction loss of {loss}")
    #     save_time(e_type,time.time() - float(event["time"]))
    elif type_log == "vectorized_logs":
        vectorized_logs = tokenizer.start_packer(data)
        loss = model.vae.get_loss(vectorized_logs)
        for i,l in enumerate(loss):
            if l > threshold:
                print(f"anomaly detected in {i} for event {id} in node {id_node} with a reconstruction loss of {l}")
    else:
        raise ValueError("operation mode not recognized")
    
    print(f"log processed in {i} for event {id} in node {id_node}")

    with open(permanent_folder+"time.txt","a") as f:
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(id_node,id,type_log,log_creation_time,catch_time,time_after_preprocess,server_catch_time,time.time(),data_size,id_server))


def reading_daemon():
    print("starting waiting for logs")
    while True:
        message = socket.recv()
        
        server_catch_time = time.time()

        data_size = sys.getsizeof(message)
        
        event = msgpack.unpackb(message)

        processing_thread = threading.Thread(target=process_event,args=(event,server_catch_time,data_size))

        processing_thread.start()


permanent_folder = "var/log/pv/logging_data/"

id_server = os.environ["HOSTNAME"].split("-")[-1]

try:
    os.mkdir(permanent_folder)
except:
    pass

vocab_size = 5000
max_len=85
latent_dim=max_len//2
threshold = 530
with open("./app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/15")

with open(permanent_folder+"time.txt","w") as f:
    f.write("{},{},{},{},{},{},{},{},{},{}\n".format("id_node","id","type","log_creation_time","catch_time","after_preprocess_time","server_catch_time","completion_time","size","id_server"))


# create an endpoint at http://localhost:/3000/

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:3000")

reader_thread = threading.Thread(target=reading_daemon,daemon=True)
reader_thread.start()

while True:
    time.sleep(10)
    print("still alive")