def map_fn(fn, *x, **kwargs):
    if len(x) == 1:
        return fn(x[0], **kwargs)
    return tuple(fn(e, **kwargs) for e in x)