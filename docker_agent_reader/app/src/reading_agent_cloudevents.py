from flask import Flask, request
import bz2
from cloudevents.http import from_http
import pickle
import sys
import os
from AI import *
import time

permanent_folder = "var/log/pv/logging_data/"
saving_data_interation = 20
try:
    os.mkdir(permanent_folder)
except:
    pass

def save_times():
    with open(permanent_folder+"times.pkl","wb") as f:
        pickle.dump(times,f)

vocab_size = 4000
moltiplicatore = 1
max_len=60*moltiplicatore # mean length + std length
latent_dim=max_len//2
threshold = 250
with open("./app/logs_tokenizer/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/")

times = {}

app = Flask(__name__)

# create an endpoint at http://localhost:/3000/
@app.route("/", methods=["POST"])
def home():
    # create a CloudEvent
    event = from_http(request.headers, request.get_data())


    # you can access cloudevent fields as seen below
    
    data = pickle.loads(bz2.decompress(event.get_data()))


    if event["type"] == "anomaly":
        times[event["id"]] = time.time() - float(event["time"])
        print(f"anomaly detected in {event['id']}")

    elif event["type"] == "logs":
        parsed_logs = tokenizer.parsing(data)
        vectorized_logs = tokenizer.vectorization(parsed_logs)
        losses = model.vae.train_step(vectorized_logs,train=False)
        if losses["reconstruction_loss"].numpy() > threshold:
            times[event["id"]] = time.time() - float(event["time"])
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {losses['reconstruction_loss'].numpy()}")
        
    elif event["type"] == "parsed_logs":
        vectorized_logs = tokenizer.vectorization(data)
        losses = model.vae.train_step(vectorized_logs,train=False)
        if losses["reconstruction_loss"].numpy() > threshold:
            times[event["id"]] = time.time() - float(event["time"])
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {losses['reconstruction_loss'].numpy()}")

    elif event["type"] == "vectorized_logs":
        losses = model.vae.train_step(data,train=False)
        if losses["reconstruction_loss"].numpy() > threshold:
            times[event["id"]] = time.time() - float(event["time"])
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {losses['reconstruction_loss'].numpy()}")

    else:
        print("error")

    
    if len(times) % saving_data_interation == 0:
        save_times()
    
    return "", 204

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=3000)




