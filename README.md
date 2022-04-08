# Learning Fast, Learning Slow 
Official Repository for Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System

We extended the [Mammoth](https://github.com/aimagelab/mammoth) framework with our method (CLS-ER) and GCIL-CIFAR-100 dataset

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

 Examples:

    ```
    python main.py --dataset seq-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-cifar10 --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-tinyimg --model clser --buffer_size 500 --load_best_args
   
    python main.py --dataset perm-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset rot-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset mnist-360 --model clser --buffer_size 500 --load_best_args```

+ For GCIL-CIFAR-100 Experiments

    `python main.py --dataset <dataset> --weight_dist <weight_dist> --model <model> --buffer_size <buffer_size> --load_best_args`

Example:

    ```
    python main.py --dataset gcil-cifar100 --weight_dist unif --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset gcil-cifar100 --weight_dist longtail --model clser --buffer_size 500 --load_best_args
    ```

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{
arani2022learning,
title={Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System},
author={Elahe Arani and Fahad Sarfraz and Bahram Zonooz},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=uxxFrDwrE7Y}
}
</pre>
