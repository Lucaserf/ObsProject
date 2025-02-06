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
dep_read["spec"]["replicas"] = 1

dep_read_doc = dep_service,dep_read


with open("./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml","w") as f:
        yaml.safe_dump_all(dep_read_doc,f)

parralel_jobs = 1
#parameters job for logs generation
dep_gen["spec"]["completions"] = parralel_jobs
dep_gen["spec"]["parallelism"] = parralel_jobs
#container 0 is the generator
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = str(time.time()) #start time
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = "100" #wait time 150, for sincronization and also waits the logging-agent to be ready
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][2]["value"] = "0.05,0.5" #period 0.2
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][3]["value"] = "16" #batch
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][4]["value"] = "42" #seed
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][5]["value"] = "BGL_created.log" #log file name (BGL_created.log,BGL.log)

#container 1 is the agent logger
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][0]["value"] = "anomaly" #operation mode (logs, vectorized_logs, anomaly)
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][1]["value"] = "False" #auto selection (True, False)
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][2]["value"] = "2000" #how many logs to send (int, inf)



with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
        yaml.dump(dep_gen,f)


subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])
subprocess.run(["kubectl","apply","-f","./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml"])

time.sleep(0) # 50 wait for the queue to be empty
subprocess.run(["kubectl","rollout","restart","deployment/dataread-deployment"])
time.sleep(20) # 20 wait for server to start before connecting the generator

#launch dummy band occupation for testing limits
subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])



