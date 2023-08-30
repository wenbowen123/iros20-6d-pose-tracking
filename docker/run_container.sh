docker rm -f se3_tracknet
DIR=$(pwd)/../
xhost + && nvidia-docker run -it --name se3_tracknet --network=host  -m  16000m  -v /usr/lib/nvidia-418:/usr/lib/nvidia-418  -v /home/bowen/debug:/home/bowen/debug -v /media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET:/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET  -v $DIR:/home/se3_tracknet -v /media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET:/DATASET -v /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/:/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/ --ipc=host -e "DISPLAY=unix:0.0" -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged -e GIT_INDEX_FILE se3_tracknet bash

