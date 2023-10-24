from flask import Flask


import time
import logging
import random
import re

app = Flask(__name__)

START = time.time()
i = 0

with open("app/quotes.txt") as f:
    list_of_quotes = f.read()


list_of_quotes = re.sub(r"\n--(.*)\n\n",r"<author>\1\t",list_of_quotes)
list_of_quotes = re.sub(r"\n",r" ",list_of_quotes)
list_of_quotes = re.sub(r"<author>",r"\n",list_of_quotes)
list_of_quotes = re.split(r"\t",list_of_quotes)

number_of_quotes = len(list_of_quotes)



def elapsed():
    global i
    running = time.time() - START
    minutes, seconds = divmod(running, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours,24)
    months, days = divmod(days,30)
    i += 1
    quote = random.choice(list_of_quotes)
    with open("/var/log/quotes{}.log".format(i),"w") as f:
        f.write("%d:%02d:%02d:%02d:%02d" % (months,days,hours, minutes, seconds))
        f.write(quote)
    return "%d:%02d:%02d:%02d:%02d" % (months,days,hours, minutes, seconds), quote



# @app.route('/')
# def root():
#     time, quote =  elapsed()
#     return render_template("app/src/template/template.html",output="Hello World! (up {})\n{}".format(time,quote))
@app.route('/')
def root():
    time, quote =  elapsed()
    output = "Hello World! (up {})\n{}".format(time,quote)
    output = re.sub(r"\n",r"<br/>",output)
    return output

@app.route('/robots.txt')
def robots():
    return "robots"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)