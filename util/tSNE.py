
import numpy as np

class TSNE_c():
    def __init__(self, **kwargs):
        self.in_path= kwargs['in_path']

    def load_data(self):
        """
        Load previously save numpy dataset
        """
        data = np.load(self.in_path+'.npy')
        labels = np.load(self.in_path+'_lbs.npy')
        return data, labels

    def ai2r_tSNE(self):
        """
        Runs tSNE on a dataset specified through the input (-d) argument
        """

        from sklearn.manifold import TSNE
        import pandas as pd
        import os
        import time
        out_dir = 'out/tSNE/test'
        if os.path.isdir(out_dir) == False:
            os.makedirs(out_dir)

        data, labels  = self.load_data()

        # Vectorize the inputs
        #reshape_size = data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]
        reshape_size = (data.shape[0], -1)
        data = np.reshape(data, reshape_size)

        # Create dataframe
        X = data
        y = labels


        feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
        df = pd.DataFrame(X,columns=feat_cols)
        df['label'] = y
        df['label'] = df['label'].apply(lambda i: str(i))
        rndperm = np.random.permutation(df.shape[0])

        time_start = time.time()
        n_sne = df.shape[0]
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
        time_end = time.time()
        time_took = time_end - time_start
        print('>ia> tSNE took {} seconds to complete'.format(time_took))

        tsne = df.loc[rndperm[:n_sne],:].copy()
        tsne['x-tsne'] = tsne_results[:,0]
        tsne['y-tsne'] = tsne_results[:,1]
        return tsne
