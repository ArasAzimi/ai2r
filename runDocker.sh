# DEFAULT: bash runDocker.sh             ---> Predict on GPU
#          bash runDocker.sh CPU         ---> Predict on CPU
#          bash runDocker.sh CPU  train  ---> Train on CPU
#          bash runDocker.sh GPU  train  ---> Train on GPU

ai2rDIR=$(pwd)
echo ">ia> Current Directory:" $ai2rDIR
echo ">ia> OS type:" $OSTYPE
if [ "$OSTYPE" == "msys" ]
then
    echo ">ia> Windows host uses CPU docker version."
    # Force to use CPU version of docker 
    if [ "$2" = "train" ]
    then
        echo ">ia> Using Docker with CPU for training."
        docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash Train.sh
    else
        echo ">ia> Using Docker with CPU for prediction."
        docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash Predict.sh
    fi
fi

if [ "$OSTYPE" == "linux-gnu" ]
then
    xhost +local:docker
    XSOCK=/tmp/.X11-unix
    XAUTH=/tmp/.docker.xauth
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -


    if [ "$1" = "cpu" ]
    then
      if [ "$2" = "train" ]
      then
        echo ">ia> Using Docker with CPU for training."
        docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash Train.sh
      else
        echo ">ia> Using Docker with CPU for prediction"
        docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash Predict.sh
      fi
    else
      if [ "$2" = "train" ]
      then
        echo ">ia> Using Docker with GPU for training."
        nvidia-docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash Train.sh
      else
            echo ">ia> DEFAULT: Using Docker with GPU for prediction."
        nvidia-docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash Predict.sh
      fi
    fi

    xhost -local:docker
fi
