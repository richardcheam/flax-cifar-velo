import collections
from functools import partial
from typing import Callable, Tuple

from absl import app
from absl import flags
from flax import linen as nn
from jax import numpy as jnp
from jax import random
from jax import tree_util
import jaxopt
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import tqdm

import models.wide_resnet
import util

import sys
sys.path.append('../')

import optuna
from optuna.pruners import MedianPruner
import jax
from optuna.pruners import SuccessiveHalvingPruner

from VeLO_training.config.hparams import suggest_hparams
from VeLO_training.config.optimizer import build_optimizer
import argparse
from flax import serialization
import pickle

Dataset = torch.utils.data.Dataset
ModuleDef = Callable[..., nn.Module]

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--opt", default="adam")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_trials", type=int, default=5)
    return p.parse_args()
    
args = parse_args()
DATASET = args.dataset
OPT = args.opt
N_TRIALS = args.n_trials
EPOCHS = args.epochs
MODEL = "wrn28_8"
SEED = 42

def save_checkpoint(params, batch_stats, path):
    to_save = {"params": params, "batch_stats": batch_stats}
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(to_save))
    print(f"Checkpoint is written to {path}")

def setup_data() -> Tuple[int, Tuple[int, int, int], Dataset, Dataset]:
    num_classes = 10
    input_shape = (32, 32, 3)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    train_dataset = datasets.CIFAR10(
        f"path/to/{DATASET}", train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(
        f"path/to/{DATASET}", train=False, download=True, transform=transform_eval)
    return num_classes, input_shape, train_dataset, val_dataset
    
def setup_data() -> Tuple[int, Tuple[int, int, int], Dataset, Dataset]:
    num_classes = 10
    input_shape = (32, 32, 3)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])

    # Load the full training dataset
    full_train_dataset = datasets.CIFAR10(
        f"path/to/{DATASET}", train=True, download=True, transform=transform_train)

    # Split 90% for training, 10% for validation
    train_size = int(0.9 * len(full_train_dataset))  # 45000
    val_size = len(full_train_dataset) - train_size  # 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    # Apply eval transform to val set (replacing training transform)
    val_dataset.dataset.transform = transform_eval
    return num_classes, input_shape, train_dataset, val_dataset


Dataset = torch.utils.data.Dataset
ModuleDef = Callable[..., nn.Module]
def make_model(num_classes: int, input_shape: Tuple[int, int, int], norm: ModuleDef = nn.BatchNorm) -> nn.Module:
    return models.wide_resnet.WideResNet(depth=28, width=8, num_classes=num_classes, norm=norm)

def train(seed, hparams):
    num_classes, input_shape, train_dataset, val_dataset = setup_data()
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
        prefetch_factor=2)
        
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
        prefetch_factor=2)

    norm = nn.BatchNorm
    norm_kwargs = lambda train: {'use_running_average': not train}

    model = make_model(num_classes, input_shape, norm=norm)
    rng_init, _ = random.split(random.PRNGKey(seed))
    init_vars = model.init(rng_init, jnp.zeros((1,) + input_shape), norm_kwargs=norm_kwargs(train=True))
    params, batch_stats = init_vars['params'], init_vars['batch_stats']

    def filter_kernel_params(tree):
        return [x for path, x in util.dict_tree_items(tree) if path[-1] == 'kernel']

    print('total number of params:',
          tree_util.tree_reduce(np.add, tree_util.tree_map(lambda x: np.prod(x.shape), params)))
    print('number of linear layers:', sum(1 for _ in filter_kernel_params(params)))

    total_steps = EPOCHS * len(train_loader)
    tx = build_optimizer(opt=OPT, hparams=hparams, num_steps=total_steps)
    opt_state = tx.init(params)

    loss_with_logits = jax.vmap(jaxopt.loss.multiclass_logistic_loss)
    
    weight_decay_vars = "all"
    weight_decay = 0.0003

    def objective_fn(params, mutable_vars, data):
        inputs, labels = data
        model_vars = {'params': params, **mutable_vars}
        outputs, mutated_vars = model.apply(
            model_vars, inputs, norm_kwargs=norm_kwargs(train=True),
            mutable=list(mutable_vars.keys()))
        example_loss = loss_with_logits(labels, outputs)
        data_loss = jnp.mean(example_loss)
        if weight_decay_vars == 'all':
            wd_vars = list(tree_util.tree_leaves(params))
        elif weight_decay_vars == 'kernel':
            wd_vars = filter_kernel_params(params)
        else:
            raise ValueError('unknown variable collection', weight_decay_vars)
        wd_loss = 0.5 * sum(jnp.sum(jnp.square(x)) for x in wd_vars)
        objective = data_loss + weight_decay * wd_loss
        return objective, (outputs, mutated_vars)

    @jax.jit
    def train_step(opt_state, params, mutable_vars, data):
        objective_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)
        (objective, aux), grads = objective_and_grad_fn(params, mutable_vars, data)
        outputs, mutated_vars = aux
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, mutated_vars, objective, outputs

    @jax.jit
    def apply_model(params, batch_stats, inputs):
        return model.apply(
            {'params': params, 'batch_stats': batch_stats}, inputs,
            norm_kwargs=norm_kwargs(train=False))

    # For per-step logging across all epochs
    # Global step counter
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(EPOCHS + 1):
        metrics = {}

        if epoch > 0:
            train_outputs = collections.defaultdict(list)
            for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_loader, f'train epoch {epoch}')):

                # Prepare input tensors
                inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
                inputs = jnp.moveaxis(inputs, -3, -1)

                # Training step
                opt_state, params, mutated_vars, objective, logits = train_step(
                    opt_state, params, {'batch_stats': batch_stats}, (inputs, labels))
                batch_stats = mutated_vars['batch_stats']

                # Compute metrics
                loss = loss_with_logits(labels, logits)
                pred = jnp.argmax(logits, axis=-1)
                acc = (pred == labels)

                # Append to epoch metrics
                train_outputs['acc'].append(acc)
                train_outputs['loss'].append(loss)
                train_outputs['objective'].append([objective])

            # Epoch-level metrics
            train_outputs = {k: np.concatenate(v) for k, v in train_outputs.items()}
            metrics.update({
                'train_loss': np.mean(train_outputs['loss']),
                'train_acc': np.mean(train_outputs['acc']),
                'train_objective': np.mean(train_outputs['objective']),
            })

        # Validation loop
        val_outputs = collections.defaultdict(list)
        for inputs, labels in tqdm.tqdm(val_loader, f'val epoch {epoch}'):
            inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
            inputs = jnp.moveaxis(inputs, -3, -1)
            logits = apply_model(params, batch_stats, inputs)
            loss = loss_with_logits(labels, logits)
            pred = jnp.argmax(logits, axis=-1)
            acc = (pred == labels)
            val_outputs['acc'].append(acc)
            val_outputs['loss'].append(loss)

        val_outputs = {k: np.concatenate(v) for k, v in val_outputs.items()}
        metrics.update({
            'val_loss': np.mean(val_outputs['loss']),
            'val_acc': np.mean(val_outputs['acc']),
        })
            
        if epoch > 0:
            train_acc_list.append(metrics['train_acc'])
            train_loss_list.append(metrics['train_loss'])
            val_acc_list.append(metrics['val_acc'])
            val_loss_list.append(metrics['val_loss'])

        # Print progress
        if epoch == 0:
            print('epoch {:d} val_acc {:.2%}'.format(epoch, metrics['val_acc']))
        else:
            print('epoch {:d}: val_acc {:.2%}, train_objective {:.6g}'.format(
                epoch, metrics['val_acc'], metrics['train_objective']))
            
    results = {
        "train_acc": train_acc_list,
        "train_loss": train_loss_list,
        "val_acc": val_acc_list,
        "val_loss": val_loss_list,
        "params": params,
        "batch_stats": batch_stats
    }  
    
    return results

def objective(trial):
    # define search space in config/hparams.py for input opt name
    HPARAMS = suggest_hparams(trial, OPT)

    # use a unique PRNGKey per trial
    KEY = SEED + trial.number
    
    results = train(seed=KEY, hparams=HPARAMS)
    # choose a metric to optimize (objective), here, maximize final validation acc
    metric = results["val_acc"][-1] #last val acc
    
    filename = f"{OPT}_{MODEL}_{DATASET}_trial_{trial.number}.msgpack"
    ckpt_path = f"hparams_ckpt/{filename}"
    
    save_checkpoint(params=results["params"],
                    batch_stats=results["batch_stats"],
                    path=ckpt_path)	
    
    # store ONLY LIGHT metadata in the study
    trial.set_user_attr("ckpt_path", ckpt_path) 
                   
    return metric 
    
################## RUN ######################

study = optuna.create_study(
    study_name=f"{OPT}_{MODEL}_{DATASET}",
    direction="maximize",
    storage=f"sqlite:///study/{OPT}_{MODEL}_{DATASET}.db", #store lightweight info from study
    load_if_exists=True,
    pruner=SuccessiveHalvingPruner()
)
study.optimize(objective, n_trials=N_TRIALS, timeout=None) 

best_trial = study.best_trial
print(f"\nBest hyperparameters: {study.best_params}\n\nsave Best HPs to path: study/{OPT}_{MODEL}_{DATASET}.db")
print("save best checkpoint to:", best_trial.user_attrs["ckpt_path"])



