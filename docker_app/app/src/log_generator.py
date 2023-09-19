import time
import pandas as pd 



df = pd.read_csv("app/OpenStack_2k.log_structured.csv")


df = df.drop(["LineId","EventId","EventTemplate"],axis=1)

def time_to_number(time):
    time = time.split(":")

    return float(time[0])*60*60+float(time[1])*60+float(time[2])

df["Pid"] = df["Pid"].apply(str)

df_time = df["Time"].apply(time_to_number)

done = False

START = time.time()
counter_prints = 0

while not done:
    mask_appened = df_time < (time.time() - START)
    logs_appened = df[mask_appened]
    df_time = df_time.drop(df_time[mask_appened].index)
    df = df.drop(logs_appened.index)

    if len(logs_appened) > 0:
        with open("/var/log/openstacklogs{}.log".format(counter_prints),"w") as f:
            for _,r in logs_appened.iterrows():
                f.write(" ".join(r)+ "\n")
        counter_prints +=1


    time.sleep(2)
    
    if not len(df)>0:
        done=True

    