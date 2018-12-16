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
def input_requrements(dataset, np_dataset):
    '''
    Checks if input data exists. This can be either raw data (images) organized
    in a directory with name as the dataset name and subdirectories with class
    names.
    '''
    import os
    if not os.path.isdir('raw/'+dataset):
        if not os.path.isfile(np_dataset+'.npy'):
            print(">ia> Input does not exists. Please either put raw data in {} or np data and labels in {}".format('raw/'+dataset, np_dataset.split(os.path.sep)[0]))
            return "input_error"
    return 'OK'
