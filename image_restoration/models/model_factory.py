from .model import MetaWeather
from easydict import EasyDict

def get_model(config, device=None):
    if config.model_name == 'metaweather':
        from .vit.config import get_config
        swin_config = get_config(EasyDict({
            'cfg': './configs/swin.yaml',
            'pretrained': './data/simmim_pretrain__swin_base__img192_window6__800ep.pth'
        }))
        mm_settings = EasyDict({
            'mode': 'topk-fixed-sponly',
            'last_dim_sp': [1024, 1024, 512, 256],
            'last_dim_ch': [7*7, 7*7, 14*14, 28*28],
            'num_shots': config.shot,
            # 'num_heads': [16, 16, 8, 4],
            # 'dim': [1024, 1024, 512, 256],
            # 'topk' : [-1, -1, -1, -1],
            # 'topk' : [1/16, 1/16, 1/32, 1/32],
            'use_residual': True
            # mm_mode=
            #      mm_heads=[8, 8, 4, 2], mm_topk=[1/4, 1/4, 1/4, 1/4], 
            #      mm_last_dim_ch=[7*7, 7*7, 14*14, 28*28], mm_last_dim_sp=[1024, 1024, 512, 256]
        })
        model = MetaWeather(config = config,
                            swin_config=swin_config, 
                            n_tasks=config.n_tasks if 'n_tasks' in config else 0,
                            num_decoders=[1, 1, 1, 1],
                            dim=[1024, 1024, 512, 256],
                            mm_settings=mm_settings)
    else:
        raise NotImplementedError(f'Invalid model \'{config.model_name}\'')
    
    if device is not None:
        model = model.to(device)
        
    return model