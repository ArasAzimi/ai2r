# DEFAULT: bash run_docker.sh             ---> Predict on GPU
#          bash run_docker.sh CPU         ---> Predict on CPU
#          bash run_docker.sh CPU  train  ---> Train on CPU
#          bash run_docker.sh GPU  train  ---> Train on GPU

ai2rDIR=$(pwd)
echo ">ia> Current Directory:" $ai2rDIR
echo ">ia> OS type:" $OSTYPE
echo $2

if [ "$OSTYPE" == "darwin18" ]
then
  echo ">ia> iOS host uses CPU docker version."
  # Force to use CPU version of docker
  if [ "$2" = "train" ]
  then
      echo ">ia> Using Docker with CPU for training."
      docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash train.sh
  else
      echo ">ia> Using Docker with CPU for prediction."
      docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash predict.sh
  fi
fi

if [ "$OSTYPE" == "msys" ]
then
    echo ">ia> Windows host uses CPU docker version."
    # Force to use CPU version of docker
    if [ "$2" = "train" ]
    then
        echo ">ia> Using Docker with CPU for training."
        docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash train.sh
    else
        echo ">ia> Using Docker with CPU for prediction."
        docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash predict.sh
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
        docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash train.sh
      else
        echo ">ia> Using Docker with CPU for prediction"
        docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0_cpu bash predict.sh
      fi
    else
      if [ "$2" = "train" ]
      then
        echo ">ia> Using Docker with GPU for training."
        nvidia-docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash train.sh
      else
            echo ">ia> DEFAULT: Using Docker with GPU for prediction."
        nvidia-docker run -it --rm --env QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash predict.sh
      fi
    fi

    xhost -local:docker
fi
