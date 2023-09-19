#! /bin/bash

docker build -t lucaserf/obs:read_log docker_app/
docker push lucaserf/obs:read_log

kubectl rollout restart deployment/dataread-deployment