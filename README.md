# AI for Airplane classification (ai2r)

### Training
Assuming that you have got the requirements setup on your local system, in a virtual environment or in a docker image; start by `python ai2r.py -d dataset_name -m model_name --gpu 1`. Alternatively you can set your desired parameters in "Train.sh" and run this bash file.
* `dataset_name` is the name of the dataset which has to be put in raw directory. If `raw` data does not exists or you want to use current ai2r dataset the datset will be downloaded and extracted in a new directory "datasets". You will see the details of this process in the prompt messages.
* `model_name` is the name of a model to be used. This should be defined in models directory. Currently this can be set to one of:  
  > inceptionv3_pretrained  
  > vgg16_pretrained  

For example: `python ai2r.py -d aircrafts -m inceptionv3_pretrained --gpu 1` will train the aircrafts dataset using inceptionv3 with trained weights from imagenet.     

If this is the first time, this command will trigger downloading the aircrafts dataset and the corresponding labels in .npy format. Once download is complete the code will unzip the data and add it to a directory i.e., "datasets". The training will start with the configured parameters in "config.json". Once the training is done the results will be saved under out directory. The results include the trained model, binarized labels in pickle format, a summary of the training and a .png graph of how the training/validation progressed in different epochs

#### Training using docker:
* `cd` to the ai2r project directory.
* Specify the model and dataset to be used in `Train.sh` script.
* Run `bash runDocker.sh gpu train`. This will run a docker container with the requirements installed. It will also mount the current directory to `/ai2r` directory in the docker so that the docker container has access to `Train.sh`. The docker image will run `Train.sh` in the same manner as if you run ai2r.py on your system or in virtual environment.
* To use a docker image with CPU version of keras/tensorflow for training use `bash runDocker.sh cpu train`.

### Prediction
Assuming some training is done and results are available under "out" directory, you can start by `python predict.py -i path_to_image`. This will scan the out directory for trained models and a prompt will ask user for the desired model to be used for prediction.

#### Prediction using docker:
* `cd` to the ai2r project directory.
* Add the path to the image you would like to run a prediction on to the `Predict.sh` script.
* Run `bash runDocker.sh`. This will run a docker container with the requirements installed. It will also mount the current directory to `/ai2r` directory in the docker so that the docker container has access to `Predict.sh`. The docker image will run `Predict.sh` in the same manner as if you run predict.py on your system or in virtual environment.
* Running `bash runDocker.sh` by default will use a GPU version docker image and run prediction on the test image specified in `Predict.sh`. To use a docker image with CPU version of keras/tensorflow for prediction use `bash runDocker.sh cpu`

### Using the Flask api
The code for the [Flask](http://flask.pocoo.org/) api is located under "api" directry. Any source code related to models training or prediction is on the root director of the project or under "src" directory. Trained models available to the api must be added to "deployment/model".
#### Testing Flask api
The required [Postman](https://www.getpostman.com/) [collection](https://github.com/ArasAzimi/ai2r/blob/master/ai2r.postman_collection.json) and [environment](https://github.com/ArasAzimi/ai2r/blob/master/ai2r.postman_environment.json) are available in the root directory of the project. These files can be imported into Postman to test or try out the api.

### Data
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
