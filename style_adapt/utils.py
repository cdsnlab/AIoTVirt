def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) >= 2)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_var(feat, eps=1e-5, to=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) >= 2)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_var = feat_var.view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    if to:
        return feat_mean.to(to), feat_var.to(to)
    return feat_mean, feat_var

def calc_mean_var_flat(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) >= 2)
    N, C = size[:2]
    feat_var = feat.reshape(1, C, -1).var(dim=2) + eps
    feat_var = feat_var.view(1, C, 1, 1)
    feat_mean = feat.reshape(1, C, -1).mean(dim=2).view(1, C, 1, 1)
    return feat_mean.cpu(), feat_var.cpu()
