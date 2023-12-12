import os
import torch
from .utils import calc_mean_var
from torch.utils import data
from tqdm import tqdm
from .config import get_opts

def extract_embeddings(model, dataset, save_path='.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
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

    save_path = os.path.join(os.path.expanduser(save_path))
    torch.save({'mean': style_means, 'std': style_vars}, os.path.join(save_path, 'style.pth'))
    torch.save({'embedding': embedding}, os.path.join(save_path, 'embedding.pth'))
