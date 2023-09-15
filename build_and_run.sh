#! /bin/bash

docker build -t lucaserf/logging_agent:latest docker_agent_logger/
docker push lucaserf/logging_agent:latest


docker build -t lucaserf/obs:gen docker_app/
docker push lucaserf/obs:gen

kubectl rollout restart deployment/server-gen-deployment
