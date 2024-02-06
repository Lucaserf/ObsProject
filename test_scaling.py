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
dep_read["spec"]["replicas"] = 2

dep_read_doc = dep_service,dep_read



with open("./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml","w") as f:
        yaml.safe_dump_all(dep_read_doc,f)


#parameters job for logs generation
dep_gen["spec"]["parallelism"] = 3
#container 0 is the generator
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = str(time.time()) #start time
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = str(120) #wait time 120, for sincronization and also waits the logging-agent to be ready
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][2]["value"] = str(0.15) #period 0.2
dep_gen["spec"]["template"]["spec"]["containers"][0]["env"][3]["value"] = str(1) #batch
#container 1 is the agent logger
dep_gen["spec"]["template"]["spec"]["containers"][1]["env"][0]["value"] = "vectorized_logs" #operation mode (logs, vectorized_logs, anomaly)



with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
        yaml.dump(dep_gen,f)


subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])
subprocess.run(["kubectl","apply","-f","./docker_agent_reader/app/deploy/logs_reader_deploy_created.yaml"])

time.sleep(100) # 100 wait for the queue to be empty
subprocess.run(["kubectl","rollout","restart","deployment/dataread-deployment"])

subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])



