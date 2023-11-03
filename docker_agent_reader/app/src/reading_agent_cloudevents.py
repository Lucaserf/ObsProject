from flask import Flask, request
import bz2
from cloudevents.http import from_http
import pickle
import sys
from AI import AnomalyDetector


latent_space_dim = 256
threshold = 10
anomaly_detector = AnomalyDetector(latent_space_dim,threshold)
app = Flask(__name__)

# create an endpoint at http://localhost:/3000/
@app.route("/", methods=["POST"])
def home():
    # create a CloudEvent
    event = from_http(request.headers, request.get_data())


    # you can access cloudevent fields as seen below
    
    data = pickle.loads(bz2.decompress(event.get_data()))
    


    print(
        f"Found {event['id']} from {event['source']} with type "
        f"{event['type']} and specversion {event['specversion']}, size {event['size']}"
        f" and data {sys.getsizeof(data)}, {data[:2]}"
    )


    loss, anomaly = anomaly_detector.detect(data)
    anomaly_detector.train_step(data)


    return f"loss: {loss}, anomaly: {anomaly}", 204


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=3000)