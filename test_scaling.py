#!/usr/bin/env python3
import yaml
import time
# import datetime
# import kubernetes as k8s
import subprocess

# k8s.config.load_kube_config()
# k8s_apps_v1  = k8s.client.AppsV1Api()
# k8s_batch_v1 = k8s.client.BatchV1Api()



with open("./docker_app/app/deploy/periodic_log_generator.yaml") as f:
    dep_gen = yaml.safe_load(f)

with open("./docker_agent_reader/app/deploy/logs_reader_deploy.yaml","r") as f:
    dep_read_doc = yaml.safe_load_all(f)
    dep_read_doc = list(dep_read_doc)
dep_service = dep_read_doc[1]
dep_read = dep_read_doc[0]
#parameters for server
dep_read["spec"]["replicas"] = 5

dep_read_doc = dep_service,dep_read



with open("./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml","w") as f:
        yaml.safe_dump_all(dep_read_doc,f)


#parameters job for logs generation
dep_gen["spec"]["parallelism"] = 5
#container 0 is the generator
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = str(time.time()) #start time
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = "150" #wait time 150, for sincronization and also waits the logging-agent to be ready
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][2]["value"] = "0.1,0.1" #period 0.2
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][3]["value"] = "1" #batch
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][4]["value"] = "42" #seed

#container 1 is the agent logger
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][0]["value"] = "logs" #operation mode (logs, vectorized_logs, anomaly)
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][1]["value"] = "False" #auto selection (True, False)
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][2]["value"] = "5000" #how many logs to send (int, inf)



with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
        yaml.dump(dep_gen,f)


subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])
subprocess.run(["kubectl","apply","-f","./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml"])

time.sleep(100) # 100 wait for the queue to be empty
subprocess.run(["kubectl","rollout","restart","deployment/dataread-deployment"])

#launch dummy band occupation for testing limits

subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])



