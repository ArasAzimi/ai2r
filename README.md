# AI for Airplane classification (ai2r)

### Training
Assuming you have got the requirements setup on your system, a virtual environment or a docker; start by `python ai2r.py -d dataset_name -m model_name --gpu 1`.
* `dataset_name` is the name of the dataset which has to be put in raw directory.
* `model_name` is the name of a model to be used. This shoud be defined in models directory. Currently this can be set to one of:  
  > inceptionv3_pretrained  
  > vgg16_pretrained  

For example: `python ai2r.py -d aircrafts -m inceptionv3_pretrained --gpu 1` will train the aircrafts dataset using inceptionv3 with trained weights from imagent.     

If this is the first time, this command will trigger downloading the aircrafts dataset and the corresponding labels in .npy format. Once download is complete the code will unzip the data and add it to a directory i.e., "datasets". The training will start with the configured parameters in config.json. Once the training is done the results will be saved under out directory. The results include the trained model, binarized labels in pickle format, a summary of the training and a .png graph of how the training/validation progressed different epochs

#### Training using docker:
* `cd` to the ai2r project directory.
* Specify the model and dataset to be used in `Train.sh` script.
* Run `bash runDocker.sh gpu train`. This will run a docker container with the requirements installed. It will also mount the current directory to `/ai2r` directory in the docker so that the docker container has access to `Train.sh`. The docker image will run `Train.sh` in the same manner as if you run ai2r.py on your system or in virtual environment.
* To use a docker image with CPU version of keras/tensorflow for training use `bash runDocker.sh cpu train`

### Prediction
Assuming some training is done and results are available under out directory, you can start by `python predict.py -i path_to_image`. This will scan the out directory for trained models and a prompt will ask user for the desired model to be used for prediction.

#### Prediction using docker:
* `cd` to the ai2r project directory.
* Add the path to the image you would like to run a prediction on to the `Predict.sh` script.
* Run `bash runDocker.sh`. This will run a docker container with the requirements installed. It will also mount the current directory to `/ai2r` directory in the docker so that the docker container has access to `Predict.sh`. The docker image will run `Predict.sh` in the same manner as if you run predict.py on your system or in virtual environment.
* Running `bash runDocker.sh` by default will use a GPU version docker image and run prediction on the test image specified in `Predict.sh`. To use a docker image with CPU version of keras/tensorflow for prediction use `bash runDocker.sh cpu`

## Data
The data is scraped from the web. Currently it contains samples for 13 types of airplanes mostly airlines. Data is not exactly balanced but it is good enough to fly!
Here is a list of airplanes in the data:  

|Type|Manufacturer|Code|#Samples|
|---|---|---|---|  
|A330|Airbus|a330|471|
|A380|Airbus|a380|747|
|B737|Boeing|b737|463|
|B757|Boeing|b757|684|
|Citation II|Cessna|c550|421|
|concord|Concord|conc|325|
|DC-10|Douglas|dc10|573|
|50|Fokker|f50|523|
|100|Fokker|f100|482|
|Falcon 900|Dassault|f900|613|
|Gulfstream V|glf5|Gulfstream|466|
|Jetstream 31|British Aerospace|js31|525|
|Yak-40|Yakovlev|yk40|531|


## Requirements:
Refer to requirements.txt
