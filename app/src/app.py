import time
from flask import Flask
app = Flask(__name__)

START = time.time()

def elapsed():
    print("calculating times")
    running = time.time() - START
    minutes, seconds = divmod(running, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours,24)
    months, days = divmod(days,30)
    print("done")
    return "%d:%02d:%02d:%02d:%02d" % (months,days,hours, minutes, seconds)

@app.route('/')
def root():
    return "Hello World (Python)! (up %s)\n" % elapsed()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)