#! /bin/bash
cd ~/ObsProject/


docker build -t lucaserf/push:latest ./traffic_gen/
docker push lucaserf/push:latest

#jobs
# kubectl apply -f ./traffic_gen/app/deploy/deploy_push.yaml
# kubectl delete -f ./traffic_gen/app/deploy/deploy_push.yaml