# This repo was adapted to train CIFAR with VeLO

### More information available [here](https://github.com/richardcheam/FLAX-VeLO) !

### Usage

```bash
python main.py --dataset_root=path/to/cifar10 --config=configs/default.py --config.arch=resnet_v1_18
```


### Dependencies

```bash
absl-py
flax
jax[cuda]
jaxopt
ml_collections
numpy 
torch
torchvision
tqdm
```
To install jax with CUDA, refer to the [installation instructions](https://github.com/google/jax#installation).
