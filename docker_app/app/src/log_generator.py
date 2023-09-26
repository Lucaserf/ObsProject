import time
import pandas as pd 


#06:25:02.870 openstack_normal1.log


def time_to_number(time):
    time = time.split(":")
    speed_up = 325
    return (float(time[0])*60*60+float(time[1])*60+float(time[2]))/speed_up 


with open("app/data/openstack_normal1.log") as f:
    logs = f.read().split("\n")[:-1]

df = pd.DataFrame(logs,columns=["Log"])

df["Time"] = df["Log"].apply(lambda x: time_to_number(x.split(" ")[2]))

done = False

START = time.time()
counter_prints = 0

while not done:
    mask_appened = df["Time"] < (time.time() - START)
    logs_appened = df[mask_appened]
    df = df.drop(logs_appened.index)

    if len(logs_appened) > 0:
        with open("/var/log/openstacklogs{}.log".format(counter_prints),"w") as f:
            for r in logs_appened["Log"]:
                f.write(r+"\n")
        counter_prints +=1


    time.sleep(0.01)
    
    if not len(df)>0:
        done=True

print("the dataset is finished")