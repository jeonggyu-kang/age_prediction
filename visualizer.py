import random

import cv2
import torch
import numpy as np

from matplotlib import pyplot as plt


try:
    import umap
except ImportError as e:
    print('{e}: Please install umap by following command: pip install umap-learn')
    exit(1)


class Umap:
    def __init__(self, 
        n_samples = 1000,
        min_dist=0.1,
        n_neighbors=15,
        n_components = 2,
    ):
        self.n_sample = n_samples
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric='euclidean'
        )

    def make_fig(self, embedding, label):
        """make figure from embedding and label"""

        fig = plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=4)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(5)-0.5).set_ticks(np.arange(4))
        plt.title('UMAP projection', fontsize=10);

        fig.canvas.draw()
        fig_arr = np.array(fig.canvas.renderer._renderer)
        fig_arr = cv2.cvtColor(fig_arr, cv2.COLOR_RGBA2RGB)
        fig_arr = fig_arr / 255
        fig_tensor = torch.from_numpy(fig_arr).permute(2,0,1)

        plt.close()

        return fig_tensor

    
    def sampling(self, data, label):
        if len(data) > self.n_sample:
            n = data.shape[0]
            data = data.reshape(n,-1).astype(np.float32)
            label = label.reshape(n,-1).astype(np.float32)

            idx = [random.randint(0,self.n_sample) for i in range(self.n_sample)]
            sampled_data = [data[i] for i in idx]
            sampled_label = [label[i] for i in idx]
            return sampled_data, sampled_label
        else:
            return data, label
    

    def __call__(self, input):
        '''
            Returns Umap visualized image
            args
                input
                    contains latent vectors and paired coordinate : dict
            return
                umap_img
                    visualized umap image : torch.Tensor (3,H,W)
        '''
        # data == latent code 
        data, label = input['latent_code'], input['label']
        
        # data conversion
        data = torch.cat(data).numpy()
        label = torch.cat(label).numpy()
        
        # data sampling
        sampled_data, sampled_label = self.sampling(data, label)

        sampled_label = list(map(lambda x: x.item(), sampled_label))

        if len(sampled_data.shape) == 4:
            sampled_data = sampled_data.mean(-1).mean(-1)
        
        # transform
        embedding = self.reducer.fit_transform(sampled_data)

        # label: int list 
        # make plot
        umap_img = self.make_fig(embedding, sampled_label)
        
        return umap_img #  torch.Tensor (3,H,W)