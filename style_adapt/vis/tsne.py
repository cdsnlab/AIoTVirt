from openTSNE import TSNE
import openTSNE.callbacks
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from typing import List
import itertools
from tqdm import tqdm




def visualize_tsne(features: List[List[torch.Tensor]], labels: List[str]=None, 
                   adapted_features: List[List[torch.Tensor]]=None, adapted_labels: List[str]=None,
                   points: List[torch.Tensor]=None, point_labels: List[str]=None, 
                   figsize=(10, 10), dimension=2, perplexity=30):
    tsne = TSNE(n_jobs=8, n_components=dimension, perplexity=perplexity)

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


class ProgressCallback(openTSNE.callbacks.Callback):
    def __init__(self, pbar: tqdm, step: int=1) -> None:
        super().__init__()
        self.pbar = pbar
        self.step = step

    def __call__(self, iteration, error, embedding):
        self.pbar.update(self.step)
        return False


def visualize_tsne_raw(features: np.ndarray, labels: np.ndarray, label_names: list[str]=None,
                   adapted_features: np.ndarray=None, adapted_labels: np.ndarray=None,
                   figsize=(10, 10), dimension=2, perplexity=30, legend_nrow=2):
    
    print(f'{features.shape=}, {labels.shape=}')

    with tqdm(total=750) as pbar:
        tsne = TSNE(n_jobs=8, 
                    n_components=dimension, 
                    perplexity=perplexity, 
                    callbacks_every_iters=25,
                    callbacks=ProgressCallback(pbar, 25))
        trained = tsne.fit(features)

    cluster = np.array(trained)

    print('t-SNE computed, waiting for plot...')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot() if dimension < 3 else fig.add_subplot(projection='3d')
    
    classes = np.unique(labels)
    for i in classes:
        idx = np.where(labels == i)
        ax_args = dict(
            marker = '.' if i < 10 else 'o', 
            label = i if label_names is None else label_names[i], 
            edgecolors = 'face' if i<10 else '#000000bb', 
            linewidths = 0.5
        )

        if dimension < 3:
            ax.scatter(cluster[idx, 0], cluster[idx, 1], **ax_args)
        else:
            ax.scatter(cluster[idx, 0], cluster[idx, 1] ,cluster[idx, 2], **ax_args)
            
    clst = None
    if adapted_features is not None and len(adapted_features) > 0:
        clst = trained.transform(adapted_features)

        for i in np.unique(adapted_labels):
            idx = np.where(adapted_labels == i)
            if dimension < 3:
                ax.scatter(clst[idx, 0], clst[idx, 1], marker='*', s=100, label=i if label_names is None else label_names[i])
            else:
                ax.scatter(clst[idx, 0], clst[idx, 1], clst[idx, 2], marker='*', s=100, label=i if label_names is None else label_names[i])

    ax.autoscale()

    plt.legend(loc='lower center', ncol=len(classes)//legend_nrow, bbox_to_anchor=(0.5, -0.05))
    plt.axis('off')
    plt.show()

    return cluster, clst, fig


