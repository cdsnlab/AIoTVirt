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