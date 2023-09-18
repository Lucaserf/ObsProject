#! /bin/bash

docker build -t lucaserf/logging_agent:latest docker_agent_logger/
docker push lucaserf/logging_agent:latest


docker build -t lucaserf/obs:sim_log docker_app/
docker push lucaserf/obs:sim_log

kubectl rollout restart deployment/datagen-deployment
