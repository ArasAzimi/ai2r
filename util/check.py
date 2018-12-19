from werkzeug.security import safe_str_cmp

def model_requirements(argument, img_size):
    '''
    Checks model requirement if any for the perticular model in use
    '''
    if safe_str_cmp(argument , "inceptionv3_pretrained"):
        inceptionV3_min_dim = 139
        if(img_size[0]<inceptionV3_min_dim or img_size[1]<inceptionV3_min_dim):
            print(">ia> Input size for Inception v3 should be at least {}*{}; got {}".format(inceptionV3_min_dim, inceptionV3_min_dim, img_size))
            return "model_error"
    return 'OK'
def input_requrements(dataset, np_dataset, downlaod_dataset_if_not_exists):
    '''
    Checks if input data exists. This can be either raw data (images) organized
    in a directory with name as the dataset name and subdirectories with class
    names.
    '''
    import os
    if not os.path.isdir('raw/'+dataset):
        if not os.path.isfile(np_dataset+'.npy'):
            if not downlaod_dataset_if_not_exists:
                dataname = np_dataset.split(os.path.sep)[0]
                print(">ia> Input does not exists. Do either one of the following:\n \
                1) Add raw train data to {} \n \
                2) Add npy train data and labels to /{} ({}.npy, {}_lbs.npy) \n \
                3) Set downlaod_ai2r_dataset in config.json to true. ".format('raw/'+dataset, dataname, dataset, dataset))
                return "input_error"
    return 'OK'
