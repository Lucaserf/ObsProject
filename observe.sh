#! /bin/bash

cd ~/ObsProject/

# kubectl get pods --no-headers -o custom-columns=":metadata.name"

pods_names=$(kubectl get pods --no-headers -o custom-columns=":metadata.name")


stringarray=($pods_names)


tmux split-window -h -p 50
tmux send-keys "watch kubectl logs ${stringarray[0]} -c logging-agent | tail -n 20" "c-m"

tmux split-window -h -p 50
tmux send-keys "watch kubectl logs ${stringarray[1]} | tail -n 20" "c-m"