import os
import torch
from .utils import calc_mean_var
from torch.utils import data
import numpy as np
from tqdm import tqdm

def extract_embeddings(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    cur_itrs = 0
    model.eval()

    """
        Style extraction
    """


    style_means = [None, None, None, None]
    style_vars = [None, None, None, None]
    embedding = []

    for i,(img_id, tar_id, images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device, dtype=torch.float32)
        for j in [0, 1, 2, 3]:
            ft = model.get_feature(images, level=j)
            mean, var = calc_mean_var(ft)
            style_means[j] = (style_means[j] * i + mean) / (i+1) if style_means[j] is not None else mean
            style_vars[j] = (style_vars[j] * i + var) / (i+1) if style_vars[j] is not None else var

    for i, s in enumerate(style_vars):
        style_vars[i] = s.sqrt()

    root_dir = os.path.join(os.path.expanduser('~/workspace/CVPR2024'))
    torch.save({'mean': style_means, 'std': style_vars}, f'{root_dir}/data/embeddings/style.pth')
    torch.save({'embedding': embedding}, f'{root_dir}/data/embeddings/embedding.pth')

if __name__ == '__main__':
    with torch.no_grad():
        extract_embeddings()