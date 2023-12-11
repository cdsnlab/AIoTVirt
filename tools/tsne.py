from openTSNE import TSNE
import openTSNE.callbacks
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from typing import List
import itertools
from tqdm import tqdm




def visualize_tsne(features: List[List[torch.Tensor]], labels: List[str]=None, 
                   adapted_features: List[List[torch.Tensor]]=None, adapted_labels: List[str]=None,
                   points: List[torch.Tensor]=None, point_labels: List[str]=None, 
                   figsize=(10, 10), dimension=2, perplexity=30):
    tsne = TSNE(n_jobs=8, n_components=dimension, perplexity=perplexity)

    # fts = [a + b for a, b in zip(features, points)]
    fts = features
    lengths = list(map(len, fts))
    try:
        fts = np.array(list(itertools.chain.from_iterable(fts)))
    except:
        fts = torch.stack(list(itertools.chain.from_iterable(fts))).numpy()
    lbs = np.array(list(itertools.chain.from_iterable([[i] * l for i, l in enumerate(lengths)])))
    print(f'{fts.shape=}, {lbs.shape=}, {lengths=}')

    trained = tsne.fit(fts)
    cluster = np.array(trained)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot() if dimension < 3 else fig.add_subplot(projection='3d')

    if labels is None:
        labels = list(range(len(features)))
    
    for i in range(len(features)):
        idx = np.where(lbs == i)
        if dimension < 3:
            ax.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=labels[i])
        else:
            ax.scatter(cluster[idx, 0], cluster[idx, 1] ,cluster[idx, 2], marker='.', label=labels[i])
            
    clst = None
    if adapted_features is not None and len(adapted_features) > 0:
        lengths_ad = list(map(len, adapted_features))
        fts_ad = np.array(list(itertools.chain.from_iterable(adapted_features)))
        lbs_ad = np.array(list(itertools.chain.from_iterable([[i] * l for i, l in enumerate(lengths_ad)])))
        clst = trained.transform(fts_ad)

        print(f'{lengths_ad=} {fts_ad.shape=} {lbs_ad.shape=} {clst.shape=}')

        for i in range(len(adapted_features)):
            idx = np.where(lbs_ad == i)
            if dimension < 3:
                ax.scatter(clst[idx, 0], clst[idx, 1], marker='.', s=100, label=adapted_labels[i])
            else:
                ax.scatter(clst[idx, 0], clst[idx, 1], clst[idx, 2], marker='.', s=100, label=adapted_labels[i])

    ax.autoscale()
    plt.legend()
    plt.show()

    return cluster, clst, fig
