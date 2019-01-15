"""
================================
SVM Standalone
================================

Run SVM on a dataset. Dataset should be put in ./test directory.
"""

from util.prep import *
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import pandas as pd
import os
import time
from werkzeug.security import safe_str_cmp

def ai2r_svm(in_data_path, img_size):
    """
    Runs tSNE on a dataset in /test directory
    """
    out_dir = 'out/svm/test'
    if os.path.isdir(out_dir) == False:
        os.makedirs(out_dir)
    if os.path.isfile(out_dir+'.npy')== False:
        time_start = time.time()
        data, labels = prepare_input(in_data_path, img_size)
        data, labels = convert_np(data, labels)
        data = normalize_input(data)
        time_end = time.time()
        time_took = time_end - time_start
        print('>ia> Prep took {} seconds to complete'.format(time_took))
        np.save(out_dir, data)
        np.save(out_dir+'_lbs', labels)
    else:
        print(">ia> Loading previously processed input...")
        data = np.load(out_dir+'.npy')
        labels = np.load(out_dir+'_lbs.npy')

    # Vectorize the inputs
    #reshape_size = data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]
    reshape_size = (data.shape[0], -1)
    print(data.shape)
    data = np.reshape(data, reshape_size)

    # Create dataframe
    X = data
    y = labels


    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))
    rndperm = np.random.permutation(df.shape[0])

    time_start = time.time()

    test_size = 0.5
    train_X, test_X, train_y, test_y = split_dataset(test_size, X, y)

    # SVM classifier
    clf = svm.SVC(gamma=0.001)
    clf.fit(train_X, train_y)

    # now to Now predict the value of the digit on the test data
    y_pred = clf.predict(test_X)

    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(test_y, y_pred)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_y, y_pred))


if __name__ == "__main__":
    from imutils import paths
    input_path='test'
    in_data_path = sorted(list(paths.list_images(input_path)))
    img_w = 75
    img_l = 75
    img_size = img_w, img_l
    if os.path.isdir(input_path) == False:
        print("Input does not exists ....")
    else:
        ai2r_svm(in_data_path, img_size)
