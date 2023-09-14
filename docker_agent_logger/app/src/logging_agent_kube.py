import time
import os
import shutil

root = "/"
log_folder = "/var/log/"
permanent_folder = "var/log/pv/logging_data"

os.mkdir(permanent_folder)


while True:
    
    data = os.listdir("/var/log/")
    print(data)
    #filtering

    data = [x for x in data if "time" in x]

    print(data)

    #log rotation
    for d in data:
        data_path = os.path.join(log_folder,d)
        shutil.move(data_path,permanent_folder)


    if len(data)> 0:
        with open("/var/log/data_logging_agent.log","a") as f:
            f.write("data rotated:\n")
            f.write(str(data))

        with open("/var/log/pv/data_logging_agent.log","a") as f:
            f.write("data rotated:\n")
            f.write(str(data))

    
    time.sleep(10)