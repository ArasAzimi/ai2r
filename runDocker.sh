# DEFAULT: bash runDocker.sh             ---> Predict on GPU
#          bash runDocker.sh CPU         ---> Predict on CPU
#          bash runDocker.sh CPU  train  ---> Train on CPU
#          bash runDocker.sh GPU  train  ---> Train on GPU

ai2rDIR=$(pwd)
echo "Current Directory:" $ai2rDIR.

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -


if [ "$1" = "cpu" ]
then
  if [ "$2" = "train" ]
  then
    echo "--- Using Docker with CPU for training---"
    docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash
  else
    echo "--- Using Docker with CPU for prediction---"
    docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu
  fi
else
  if [ "$2" = "train" ]
  then
    echo "--- Using Docker with GPU for training---"
    docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash
  else
        echo "--- DEFAULT: Using Docker with GPU for prediction---"
    docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0
  fi
fi

xhost -local:docker
