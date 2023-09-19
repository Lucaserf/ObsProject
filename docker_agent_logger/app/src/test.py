import sys
sys.dont_write_bytecode = True
import pandas as pd
import pickle
import bz2

from pre_processing import Tokenizer



df = pd.read_csv("docker_agent_logger/app/src/OpenStack_2k.log_structured.csv")
df = df.drop(["LineId","EventId","EventTemplate"],axis=1)
def time_to_number(time):
    time = time.split(":")
    return float(time[0])*60*60+float(time[1])*60+float(time[2])

df["Pid"] = df["Pid"].apply(str)

df_time = df["Time"].apply(time_to_number)


logs = []

for i,r in df.iterrows():
    logs.append(" ".join(r))

tokenizer = Tokenizer()

reduced_logs = logs

# print(reduced_logs)

with open("data.log","a") as f:
    f.write("\n".join(reduced_logs)+ "\n")

with bz2.open("pickled_data_bz2.log","wb") as f:
    pickle.dump(reduced_logs,f)



encoded_data = tokenizer.preprocess(reduced_logs)
# print(encoded_data)

# print(tokenizer.decode(encoded_data))
with bz2.open("encoded_data.log","wb") as f:
    pickle.dump(encoded_data,f)


for log in encoded_data:
    with bz2.open("encoded_data_separated.log","ab") as f:
        pickle.dump(log,f)

tokenizer.training_tokenizer(logs)

encoded_data = tokenizer.preprocess(reduced_logs)
# print(encoded_data)

# print(tokenizer.decode(encoded_data))

with bz2.open("trained_encoded_data.log","wb") as f:
    pickle.dump(encoded_data,f)

# logs = []
# for log in encoded_data:
#     try:
#         with bz2.open("trained_encoded_data_separated.log","rb") as f:
#             logs = pickle.load(f)
#         logs.append(log)
#     except:
#         logs.append(log)

#     with bz2.open("trained_encoded_data_separated.log","wb") as f:
#         pickle.dump(logs,f)
