import sys
sys.dont_write_bytecode = True
from AI import *
import os
import drain3
from drain3.file_persistence import FilePersistence

os.chdir("./docker_agent_logger/app/")

with open("./data/openstack_normal1.log") as f:
    logs = f.read().split("\n")[:-1]

persistence = FilePersistence("./drain/drain3_state.bin")


config = drain3.template_miner_config.TemplateMinerConfig()
config.load("./drain/drain3.ini")

template_miner = drain3.TemplateMiner(persistence,config)



for log in logs[:10]:
    result = template_miner.add_log_message(log)
    print(result)


# tokenizer = Tokenizer("./bert-base-cased_saved", max_len=512)

# dataset = (i for i in logs)

# tokenizer.training_tokenizer(dataset,vocab_size = 32000)

# tokenizer.saving_tokenizer("./logs_tokenizer")
