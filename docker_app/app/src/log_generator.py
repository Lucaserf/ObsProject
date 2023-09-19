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
i = 0

while not done:
    time = time.time() - START
    print(len(df))
    print(len(df_time))
    logs_appened = df[df_time < time]
    df = df.drop(logs_appened.index)


    with open("/var/log/openstacklogs{}.log".format(i),"w") as f:
        for i,r in logs_appened.iterrows():
            f.write(" ".join(r))


    time.sleep(2)
    i +=1
    
    if not len(df)>0:
        done=True