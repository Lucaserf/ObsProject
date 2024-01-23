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
    dep = yaml.safe_load(f)

# i = 1
# scaling = 1

# while i < 5:

#     dep["spec"]["parallelism"] = i

#     with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
#         yaml.dump(dep,f)

#     subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])

#     print("job updated with parallelism: {}".format(i))

#     # run for x minutes
#     time.sleep(1*60)

#     i += scaling


# subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])
    

subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])

time.sleep(100) # wait for the queue to be empty
subprocess.run(["kubectl","rollout","restart","deployment/dataread-deployment"])


dep["spec"]["parallelism"] = 1
#container 0 is the generator
dep["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = str(time.time()) #start time
dep["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"] = str(120) #wait time 120, works



with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
        yaml.dump(dep,f)

subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])



