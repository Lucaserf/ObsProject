#! /bin/bash

cd ~/ObsProject/

docker build -t lucaserf/reading_agent:latest docker_agent_reader/
docker push lucaserf/reading_agent:latest

kubectl rollout restart deployment/dataread-deployment

# kubectl apply -f ./docker_agent_reader/app/deploy/logs_reader-deploy.yaml