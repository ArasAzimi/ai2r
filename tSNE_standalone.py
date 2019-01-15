"""
================================
tSNE Standalone
================================

Run tSNE on a dataset specified through the input (-d) argument
Following Arguments are used to calculate TSNE:
- n_components = 2
- verbose      = 1
- perplexity   = 40
- n_iter       = 300
"""

print(__doc__)

from ggplot import *
from util import check
from util.tSNE import TSNE_c
import argparse

def main():
    from werkzeug.security import safe_str_cmp
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,	help="path to input dataset")
    args = vars(ap.parse_args())
    dataset = args["dataset"]
    datasets_dir = 'datasets'

    downlaod_dataset_if_not_exists = False
    if downlaod_dataset_if_not_exists:
	       use_raw_data = False

    #out_dirx = 'out/'+ dataset+ '/'+ model_name+ '_e'+ str(epochs)+ '_lr'+ str(learning_rate)+ '_bs'+ str(batch_size)+ '/'
    np_dataset = datasets_dir+'/'+dataset

    r = check.input_requrements(dataset, np_dataset, downlaod_dataset_if_not_exists)
    if not safe_str_cmp(r , "OK"):
    	print(">ia> Exited with error: {}".format(r))
    	return 0

    tSNE= TSNE_c(in_path=np_dataset)
    tsne = tSNE.ai2r_tSNE()
    chart = ggplot( tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
                + geom_point(size=70,alpha=0.1) \
                + ggtitle("tSNE dimensions colored by class")
    print(chart)

# Python 3 does not need if __name__ == "__main__":
if __name__ == "__main__":
    main()
