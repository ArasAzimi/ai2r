ai2rDIR=$(pwd)
echo "Current Directory:" $ai2rDIR.
# run to predict using the model specified in Predict.sh
nvidia-docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0

# run image in bash mode to train or use as you wish
#nvidia-docker run -it -v $ai2rDIR:/ai2r azmer/ai2r:rev1.0 bash
