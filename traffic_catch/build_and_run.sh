#! /bin/bash
cd ~/ObsProject/


docker build -t lucaserf/pull:latest ./traffic_catch/
docker push lucaserf/pull:latest

#jobs
# kubectl apply -f ./traffic_catch/app/deploy/deploy_pull.yaml
# kubectl delete -f ./traffic_catch/app/deploy/deploy_pull.yaml