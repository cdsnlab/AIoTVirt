import torch
from einops import rearrange, repeat


def get_reshaper(pattern):
    def reshaper(x, contiguous=False, **kwargs):
        if isinstance(x, torch.Tensor):
            x = rearrange(x, pattern, **kwargs)
            if contiguous:
                x = x.contiguous()
            return x
        elif isinstance(x, dict):
            return {key: reshaper(x[key], contiguous=contiguous, **kwargs) for key in x}
        elif isinstance(x, tuple):
            return tuple(reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x)
        elif isinstance(x, list):
            return [reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x]
        else:
            return x
    
    return reshaper


from_6d_to_4d = get_reshaper('B T N C H W -> (B T N) C H W')
from_4d_to_6d = get_reshaper('(B T N) C H W -> B T N C H W')

from_6d_to_3d = get_reshaper('B T N C H W -> (B T) (N H W) C')
from_3d_to_6d = get_reshaper('(B T) (N H W) C -> B T N C H W')

from_4d_to_3d = get_reshaper('(B T N) C H W -> (B T) (N H W) C')
from_3d_to_4d = get_reshaper('(B T) (N H W) C -> (B T N) C H W')
