import sys
sys.dont_write_bytecode = True
from AI import *
import os
# import drain3
# from drain3.file_persistence import FilePersistence

os.chdir("./docker_agent_logger/app/")

with open("./data/openstack_normal1.log") as f:
    logs = f.read().split("\n")[:-1]




logs = [re.sub(r'\b[a-zA-Z\d-_]{20,}\b', '*', log) for log in logs]
# persistence = FilePersistence("./drain/drain3_state.bin")


# config = drain3.template_miner_config.TemplateMinerConfig()
# config.load("./drain/drain3.ini")

# template_miner = drain3.TemplateMiner(persistence_handler=persistence,config=config)


# for log in logs:
#     result = template_miner.add_log_message(log)

# template_miner.save_state("done training")

# for log_line in logs[:10]:
#     cluster = template_miner.match(log_line)
#     if cluster is None:
#         print(f"No match found")
#     else:
#         template = cluster.get_template()
#         print(log_line)
#         print(f"Matched template #{cluster.cluster_id}: {template}")
#         print(f"Parameters: {template_miner.get_parameter_list(template, log_line)}")
#         print()

tokenizer = Tokenizer("./bert-base-cased_saved", max_len=512)

dataset = (i for i in logs)

tokenizer.training_tokenizer(dataset,vocab_size = 32000)

tokenizer.saving_tokenizer("./logs_tokenizer")
