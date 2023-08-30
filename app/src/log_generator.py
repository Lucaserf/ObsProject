import time


def elapsed():
    print("calculating times")
    running = time.time() - START
    minutes, seconds = divmod(running, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours,24)
    months, days = divmod(days,30)
    print("done")
    
    with open("/var/log/time{}.log".format(running),"w") as f:
        f.write("%d:%02d:%02d:%02d:%02d" % (months,days,hours, minutes, seconds))
    return "%d:%02d:%02d:%02d:%02d" % (months,days,hours, minutes, seconds)

START = time.time()

while True:
    print("Hello World! (up %s)\n" % elapsed())
    time.sleep(5)