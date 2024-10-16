The code associated to [MetaWeather: Few-shot Weather-Degraded Image Restoration](https://arxiv.org/abs/2308.14334) (ECCV 2024).

# Setup
```
    image_restoration
    ├ data
    │   ├ BID
    │   │  ├ case1
    │   │  ├ case2
    │   │  ├ ...
    │   │  ├ case6
    │   │  └ gt
    │   ├ SPAData/real_test_100
    │   │  ├ input
    │   │  └ gt
    │   ├ realsnow
    │   │  ├ train
    │   │  │  ├ input
    │   │  │  └ gt
    │   │  └ test
    │   │     ├ input
    │   │     └ gt
    │   ├ BID-case1-train.txt
    │   ├ BID-case1-test.txt
    │   ├ BID-case2-train.txt
    │   ├ BID-case2-test.txt
    │   ├ ...
    │   ├ BID-case6-test.txt
    │   ├ spadata-train.txt
    │   ├ spadata-test.txt
    │   ├ realsnow-train.txt
    │   ├ realsnow-test.txt
    │   └ simmim_pretrain__swin_base__img192_window6__800ep.pth
    └ checkpoints
        ├ metaweather_meta_train.pth
        ├ metaweather_1shot_case1.pth
        ├ metaweather_1shot_case2.pth
        ├ ...
```

## Datasets
Download the datasets below:
* BID Dataset [[Link]](https://github.com/JunlinHan/BID)
* SPA-Data [[Link]](https://github.com/stevewongv/SPANet)
* RealSnow [[Link]](https://github.com/zhuyr97/WGWS-Net)

## Pretrained MetaWeather Model
Download checkpoints [[Link]](https://drive.google.com/drive/folders/1eyeaTLQXeLREhMGYYWb9MfjYTz1K7DHG?usp=sharing)
* Place them in `./checkpoints/`

## Prerequisite
Use conda environment: 
```bash
conda env create -f env.yaml
```

## Swin Transformer
If you want to train the model from the scratch, download `simmim_pretrain__swin_base__img192_window6__800ep.pth` from [[Link]](https://github.com/microsoft/SimMIM)
* Place the file in `data`.

# Usage
## Meta-Train
```bash
python3 train.py --stage=0 --meta_train=True --exp_name=test
```

## Meta-Test
```bash
python3 train.py --stage=1 --case=<case> --meta_train=True --exp_name=test --checkpoint=<path-to-checkpoint>
```
* case: (1-6) BID Task II.A Case 1-6, (7) SPA-Data, (8) RealSnow

## Evaluation
```bash
python3 test.py --case=<case> --exp_name=test --checkpoint=<path-to-checkpoint>
```
* case: (1-6) BID Task II.A Case 1-6, (7) SPA-Data, (8) RealSnow

Examples

```bash
python3 test.py --case=1 --checkpoint=./checkpoints/metaweather_1shot_case1.pth
```

```bash
python3 test.py --case=3 --checkpoint=./checkpoints/metaweather_1shot_case3.pth
```

Tested on Ubuntu 20.04, NVIDIA RTX 3090, CUDA 11.7, PyTorch 2.0.1, Python 3.10

# References
Our code is built upon the following works:
* [Visual Token Matching](https://github.com/GitGyun/visual_token_matching)
* [Transweather](https://github.com/jeya-maria-jose/TransWeather)
* [NAFNet](https://github.com/megvii-research/NAFNet)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)