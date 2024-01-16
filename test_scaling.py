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

i = 1
scaling = 3

while i < 30:

    dep["spec"]["parallelism"] = i

    with open("./docker_app/app/deploy/periodic_log_generator_created.yaml","w") as f:
        yaml.dump(dep,f)

    subprocess.run(["kubectl","apply","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])

    print("job updated with parallelism: {}".format(i))

    # run for 5 minutes
    time.sleep(5*60)

    i += scaling


subprocess.run(["kubectl","delete","-f","./docker_app/app/deploy/periodic_log_generator_created.yaml"])