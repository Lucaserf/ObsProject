#! /bin/bash

cd ~/ObsProject/

docker build -t lucaserf/reading_agent:cloudevents docker_agent_reader/
docker push lucaserf/reading_agent:cloudevents

kubectl rollout restart deployment/dataread-deployment

# kubectl apply -f ./docker_agent_reader/app/deploy/logs_reader-deploy.yaml
# kubectl delete -f ./docker_agent_reader/app/deploy/logs_reader-deploy.yaml