from flask import Flask, request
import bz2
from cloudevents.http import from_http
import pickle
import sys
import os
from AI import *
import time

permanent_folder = "var/log/pv/logging_data/"
saving_data_interation = 50
try:
    os.mkdir(permanent_folder)
except:
    pass

def save_times():
    with open(permanent_folder+"times.pkl","wb") as f:
        pickle.dump(times,f)

vocab_size = 5000
max_len=85
latent_dim=max_len//2
threshold = 530
with open("./app/logs_tokenizer/vocab_bgl.pkl","rb") as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab,max_len=max_len)
model = Model(vocab_size = vocab_size,latent_dim=latent_dim,embedding_dim=128,max_len = max_len)
model.vae.load_model(chkpt="./app/trained_classifier/15")

times = {}

app = Flask(__name__)

# create an endpoint at http://localhost:/3000/
@app.route("/", methods=["POST"])
def home():
    # create a CloudEvent
    event = from_http(request.headers, request.get_data())

    data = pickle.loads(bz2.decompress(event.get_data()))

    id = int(event["id"])
    e_type = event["type"]


    if e_type == "anomaly":
        times[id] = time.time() - float(event["time"])
        print(f"anomaly detected in {id}")

    elif e_type == "logs":
        parsed_logs = tokenizer.parsing(data)
        vectorized_logs = tokenizer.vectorization(parsed_logs)
        loss = model.vae.get_loss(vectorized_logs)
        if loss > threshold:
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {loss}")
        times[id] = time.time() - float(event["time"])
    elif event["type"] == "parsed_logs":
        vectorized_logs = tokenizer.vectorization(data)
        loss = model.vae.get_loss(vectorized_logs)
        if loss > threshold:
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {loss}")
        times[id] = time.time() - float(event["time"])
    elif event["type"] == "vectorized_logs":
        loss = model.vae.get_loss(vectorized_logs)
        if loss > threshold:
            print(f"anomaly detected in {event['id']} with a reconstruction loss of {loss}")
        times[id] = time.time() - float(event["time"])
    else:
        print("error")

    
    if len(times) % saving_data_interation == 0:
        save_times()
    
    return "", 204

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=3000)




