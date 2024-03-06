#! /bin/bash

cd ~/ObsProject/

# docker build --no-cache -t lucaserf/logging_agent:latest ./docker_agent_logger/
docker build -t lucaserf/logging_agent:latest ./docker_agent_logger/
docker push lucaserf/logging_agent:latest


# docker build --no-cache -t lucaserf/obs:sim_log ./docker_app/
docker build -t lucaserf/obs:sim_log ./docker_app/
docker push lucaserf/obs:sim_log


kubectl rollout restart deployment/cloudevents-gen-deployment

# kubectl apply -f ./docker_app/app/deploy/cloudevents-deploy.yaml
# kubectl delete -f ./docker_app/app/deploy/cloudevents-deploy.yaml

#jobs
# kubectl apply -f ./docker_app/app/deploy/periodic_log_generator.yaml
# kubectl delete -f ./docker_app/app/deploy/periodic_log_generator.yaml