# Learning Fast, Learning Slow 
Official Repository for ICLR'22 Paper ["Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System"](https://arxiv.org/pdf/2201.12604.pdf)

![Screenshot 2023-08-07 at 10 47 11](https://github.com/NeurAI-Lab/CLS-ER/assets/27284368/d31732ca-a8f7-4785-b474-fae48a4a5a47)


We extended the [Mammoth](https://github.com/aimagelab/mammoth) framework with our method (CLS-ER) and GCIL-CIFAR-100 dataset

## Additional Results
For a more extensive evaluation of our our method and benchmarking, we evaluated CLS-ER on S-CIFAR100 with 5 Tasks and also  provide the Task-IL results for all the settings. Note that similar to DER, Task-IL results merely use logit masking at inference.

|             |   S-MNIST  |            | S-CIFAR-10 |            | S-CIFAR-100 |           |  S-TinyImg |            |
|-------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Buffer Size | Class-IL   | Task-IL    | Class-IL   | Task-IL    | Class-IL   | Task-IL    | Class-IL   | Task-IL    |
|         200 | 89.54±0.21 | 97.97±0.17 | 66.19±0.75 | 93.90±0.60 | 43.80±1.89 | 73.49±1.04 | 23.47±0.80 | 49.60±0.72 |
|         500 | 92.05±0.32 | 98.95±0.10 | 75.22±0.71 | 94.94±0.53 | 51.40±1.00 | 78.12±0.24 | 31.03±0.56 | 60.41±0.50 |
|        5120 | 95.73±0.11 | 99.40±0.04 | 86.78±0.17 | 97.08±0.09 | 65.77±0.49 | 84.46±0.45 | 46.74±0.31 | 75.81±0.35 |

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

 Examples:

    python main.py --dataset seq-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-cifar10 --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-tinyimg --model clser --buffer_size 500 --load_best_args
   
    python main.py --dataset perm-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset rot-mnist --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset mnist-360 --model clser --buffer_size 500 --load_best_args

+ For GCIL-CIFAR-100 Experiments

    `python main.py --dataset <dataset> --weight_dist <weight_dist> --model <model> --buffer_size <buffer_size> --load_best_args`

Example:

    python main.py --dataset gcil-cifar100 --weight_dist unif --model clser --buffer_size 500 --load_best_args
    
    python main.py --dataset gcil-cifar100 --weight_dist longtail --model clser --buffer_size 500 --load_best_args

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

    @inproceedings{
      arani2022learning,
      title={Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System},
      author={Elahe Arani and Fahad Sarfraz and Bahram Zonooz},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=uxxFrDwrE7Y}
    }
