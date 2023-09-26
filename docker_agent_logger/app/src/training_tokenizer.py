import sys
sys.dont_write_bytecode = True
from AI import *
import os

os.chdir("./docker_agent_logger/app/")

with open("./data/openstack_normal1.log") as f:
    logs = f.read().split("\n")[:-1]

tokenizer = Tokenizer("./bert-base-cased_saved", max_len=512)

dataset = (i for i in logs)

tokenizer.training_tokenizer(dataset,vocab_size = 32000)

tokenizer.saving_tokenizer("./logs_tokenizer")



