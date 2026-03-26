#!/bin/bash

user=$(whoami)
#container_name="${user}-cuda-dev-20-121"
container_name="${user}-cuda-dev-24-125"

has_container() {
    [ $( docker ps -a | grep $container_name | wc -l ) -gt 0 ]
}

start_container() {
    echo "Entering container..."
    docker start $container_name
    docker exec -it $container_name /bin/zsh
}

start_new_container() {
    echo "Starting a new container..."
    docker run -it \
        --runtime=nvidia --gpus=all \
        --ipc=host \
        --runtime=nvidia \
        --name $container_name \
        --privileged \
        --userns host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -v "$HOME/scallop-v2":"/home/$user/scallop-v2":Z \
        -w "/home/$user" \
        -e "TERM=xterm-256color" \
        "${user}-cuda-dev-24-125"
}
if has_container; then
    start_container
else
    start_new_container
fi

