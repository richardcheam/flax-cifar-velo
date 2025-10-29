import collections
from functools import partial
import sys
sys.path.append('../')
from typing import Callable, Tuple, NamedTuple, Any
import time

from absl import app
from absl import flags
from flax import linen as nn
import jax
from jax import numpy as jnp, random, tree_util
import jaxopt
from jaxopt._src import base, tree_util as jopt_tree_util
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import torch.utils.data
from torchvision import datasets, transforms
import tqdm

import models.densenet
import models.resnet_v1
import models.resnet_v2
import models.vgg
import models.wide_resnet
import util

from VeLO_training.config.optimizer import get_velo_optimizer

# FLAGS
flags.DEFINE_string('dataset_root', None, 'Path to data.', required=True)
flags.DEFINE_bool('download', False, 'Download dataset.')
flags.DEFINE_integer('eval_batch_size', 128, 'Batch size to use during evaluation.')
flags.DEFINE_integer('loader_num_workers', 4, 'num_workers for DataLoader')
flags.DEFINE_integer('loader_prefetch_factor', 2, 'prefetch_factor for DataLoader')
flags.DEFINE_string('optimizer', None, 'Optimizer name', required=True)
#flags.DEFINE_integer('seed', None, 'Seed number', required=True)
config_flags.DEFINE_config_file('config')

from ml_collections import config_flags
from absl import flags


FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    config_flags.DEFINE_config_file('config')

Dataset = torch.utils.data.Dataset
ModuleDef = Callable[..., nn.Module]

# Custom OptaxState and VeLO-compatible lopt_update
class OptaxState(NamedTuple):
    iter_num: int
    value: float
    error: float
    internal_state: NamedTuple
    aux: Any

def lopt_update(self, params: Any, state: NamedTuple, *args, **kwargs) -> base.OptStep:
    if self.pre_update:
        params, state = self.pre_update(params, state, *args, **kwargs)
    (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)
    delta, opt_state = self.opt.update(grad, state.internal_state, params, extra_args={"loss": value})
    params = self._apply_updates(params, delta)
    new_state = OptaxState(
        iter_num=state.iter_num + 1,
        error=jopt_tree_util.tree_l2_norm(grad),
        value=value,
        aux=aux,
        internal_state=opt_state)
    return base.OptStep(params=params, state=new_state)

def main(_):
    config = ml_collections.ConfigDict(FLAGS.config)
    num_classes, input_shape, train_dataset, val_dataset = setup_data()

    config.num_classes = num_classes  # make num_classes accessible in loss
    
    for SEED in [4]:
        print(f"######################## ruuning seed {SEED} ########################")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train.batch_size, shuffle=True,
            pin_memory=False, num_workers=FLAGS.loader_num_workers, prefetch_factor=FLAGS.loader_prefetch_factor)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=FLAGS.eval_batch_size, shuffle=False,
            pin_memory=False, num_workers=FLAGS.loader_num_workers, prefetch_factor=FLAGS.loader_prefetch_factor)

        norm = nn.BatchNorm
        norm_kwargs = lambda train: {'use_running_average': not train}
        model = make_model(config, num_classes, input_shape, norm=norm)
        rng_init, _ = random.split(random.PRNGKey(SEED))
        init_vars = model.init(rng_init, jnp.zeros((1,) + input_shape), norm_kwargs=norm_kwargs(train=True))
        params, batch_stats = init_vars['params'], init_vars['batch_stats']

        def filter_kernel_params(tree):
            return [x for path, x in util.dict_tree_items(tree) if path[-1] == 'kernel']

        total_steps = config.train.num_epochs * len(train_loader)
        # loss_with_logits = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)
        loss_with_logits = jax.vmap(optax.softmax_cross_entropy_with_integer_labels, in_axes=(0, 0))


        def objective_fn(params, mutable_vars, data):
            inputs, labels = data
            model_vars = {'params': params, **mutable_vars}
            outputs, mutated_vars = model.apply(
                model_vars, inputs, norm_kwargs=norm_kwargs(train=True),
                mutable=list(mutable_vars.keys()))
            
            outputs = outputs.astype(jnp.float32)  # <--- FIX

            data_loss = jnp.mean(loss_with_logits(outputs, labels))
            if config.train.weight_decay_vars == 'all':
                wd_vars = list(tree_util.tree_leaves(params))
            elif config.train.weight_decay_vars == 'kernel':
                wd_vars = filter_kernel_params(params)
            else:
                raise ValueError('unknown variable collection', config.train.weight_decay_vars)
            wd_loss = 0.5 * sum(jnp.sum(jnp.square(x)) for x in wd_vars)
            objective = data_loss + config.train.weight_decay * wd_loss
            return objective, (outputs, mutated_vars)

        if FLAGS.optimizer == 'velo':
            tx = get_velo_optimizer(total_steps)
            solver = jaxopt.OptaxSolver(
                opt=tx,
                fun=jax.value_and_grad(objective_fn, has_aux=True),
                has_aux=True,
                value_and_grad=True,
                maxiter=total_steps
            )
            sample_batch = next(iter(train_loader))
            inputs, labels = jnp.asarray(sample_batch[0].numpy()), jnp.asarray(sample_batch[1].numpy())
            inputs = jnp.moveaxis(inputs, -3, -1)
            # opt_state = solver.init_state(params, (inputs, labels), mutable_vars={'batch_stats': batch_stats})
            opt_state = solver.init_state(params, {'batch_stats': batch_stats}, (inputs, labels))
            velo_train_step = jax.jit(partial(lopt_update, solver))
        else:
            schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
            if FLAGS.optimizer == 'sgd':
                tx = optax.sgd(schedule)
            elif FLAGS.optimizer == 'sgdm':
                tx = optax.sgd(schedule, momentum=config.train.momentum)
            elif FLAGS.optimizer == 'adam':
                tx = optax.adam(schedule)
            elif FLAGS.optimizer == 'adamw':
                tx = optax.adamw(schedule, weight_decay=config.train.adam_weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {FLAGS.optimizer}")
            opt_state = tx.init(params)

        @jax.jit
        def train_step(opt_state, params, mutable_vars, data):
            objective_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)
            (objective, aux), grads = objective_and_grad_fn(params, mutable_vars, data)
            outputs, mutated_vars = aux
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, mutated_vars, objective, outputs

        @jax.jit
        def apply_model(params, batch_stats, inputs):
            return model.apply({'params': params, 'batch_stats': batch_stats}, inputs,
                               norm_kwargs=norm_kwargs(train=False))
                           
        total_start = time.time()
        # For per-step logging across all epochs
        # Global step counter
        step_train_acc = []
        step_train_loss = []
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        global_step = 0

        for epoch in range(config.train.num_epochs + 1):
            metrics = {}

            if epoch > 0:
                train_outputs = collections.defaultdict(list)
                #for inputs, labels in tqdm.tqdm(train_loader, f'train epoch {epoch}'):
                for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_loader, f'train epoch {epoch}')):
                    inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
                    inputs = jnp.moveaxis(inputs, -3, -1)

                    if FLAGS.optimizer == 'velo':
                        step_result = velo_train_step(
                            params=params,
                            state=opt_state,
                            data=(inputs, labels),
                            mutable_vars={'batch_stats': batch_stats})
                        params, opt_state = step_result
                        logits, mutated_vars = opt_state.aux  # ✅ Unpack tuple correctly from objective_fn
                        batch_stats = mutated_vars['batch_stats']
                        objective = opt_state.value
                    else:
                        opt_state, params, mutated_vars, objective, logits = train_step(
                            opt_state, params, {'batch_stats': batch_stats}, (inputs, labels))
                        batch_stats = mutated_vars['batch_stats']

                    loss = loss_with_logits(logits.astype(jnp.float32), labels)
                    pred = jnp.argmax(logits, axis=-1)
                    acc = (pred == labels)
                    
                    train_outputs['acc'].append(acc)
                    train_outputs['loss'].append(loss)
                    train_outputs['objective'].append([objective])
                    
                     # Append per-step values
                    step_train_acc.append(float(jnp.mean(acc)))
                    step_train_loss.append(float(jnp.mean(loss)))
                    
                    global_step += 1
            
                train_outputs = {k: np.concatenate(v) for k, v in train_outputs.items()}
                metrics.update({
                    'train_loss': np.mean(train_outputs['loss']),
                    'train_acc': np.mean(train_outputs['acc']),
                    'train_objective': np.mean(train_outputs['objective']),
                })

            val_outputs = collections.defaultdict(list)
            for inputs, labels in tqdm.tqdm(val_loader, f'val epoch {epoch}'):
                inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
                inputs = jnp.moveaxis(inputs, -3, -1)
                logits = apply_model(params, batch_stats, inputs)

                logits = logits.astype(jnp.float32)  # ✅ Explicit cast here
                loss = loss_with_logits(logits, labels)  # ✅ Correct order
            

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

            if epoch == 0:
                print(f'epoch {epoch:2d}: val_acc {metrics["val_acc"]:.2%}')
            else:
                print(f'epoch {epoch:2d}: val_acc {metrics["val_acc"]:.2%}, train_obj {metrics["train_objective"]:.6g}')
            
        total_time = time.time() - total_start
        print(f"Total training time: {total_time:.2f}s")
        
        results = {
            "train_acc": train_acc_list,
            "train_loss": train_loss_list,
            "val_acc": val_acc_list,
            "val_loss": val_loss_list,
            "step_train_acc": step_train_acc,
            "step_train_loss": step_train_loss,
            "params": params,
            "batch_stats": batch_stats,
            "num_steps": total_steps
        }   

        import pickle
        # with open(f"results/metrics/{MODEL}_{DATASET}_{FLAGS.optimizer}_{FLAGS.SEED}.pkl"), "wb") as f:
        with open(f"metrics/{config.arch}_cifar10_{FLAGS.optimizer}_seed{SEED}.pkl", "wb") as f:
            pickle.dump(results, f)

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
        FLAGS.dataset_root, train=True, download=FLAGS.download, transform=transform_train)
    val_dataset = datasets.CIFAR10(
        FLAGS.dataset_root, train=False, download=FLAGS.download, transform=transform_eval)
    return num_classes, input_shape, train_dataset, val_dataset

def make_model(config: ml_collections.ConfigDict, num_classes: int, input_shape: Tuple[int, int, int], norm: ModuleDef = nn.BatchNorm) -> nn.Module:
    try:
        model_fn = {
            'resnet_v1_18': partial(models.resnet_v1.ResNet18, stem_variant='cifar'),
            'resnet_v1_34': partial(models.resnet_v1.ResNet34, stem_variant='cifar'),
            'resnet_v1_50': partial(models.resnet_v1.ResNet50, stem_variant='cifar'),
            'resnet_v2_18': partial(models.resnet_v2.ResNet18, stem_variant='cifar'),
            'resnet_v2_34': partial(models.resnet_v2.ResNet34, stem_variant='cifar'),
            'resnet_v2_50': partial(models.resnet_v2.ResNet50, stem_variant='cifar'),
            'wrn28_2': partial(models.wide_resnet.WideResNet, depth=28, width=2),
            'wrn28_8': partial(models.wide_resnet.WideResNet, depth=28, width=8),
            'densenet121_12': models.densenet.densenet_cifar,
            'densenet121_32': models.densenet.DenseNet121,
            'densenet169_32': models.densenet.DenseNet169,
            'densenet201_32': models.densenet.DenseNet201,
            'densenet161_48': models.densenet.DenseNet161,
            'vgg11_backbone': models.vgg.VGG11Backbone,
            'vgg13_backbone': models.vgg.VGG13Backbone,
            'vgg16_backbone': models.vgg.VGG16Backbone,
            'vgg19_backbone': models.vgg.VGG19Backbone,
            'vgg11': models.vgg.VGG11,
            'vgg13': models.vgg.VGG13,
            'vgg16': models.vgg.VGG16,
            'vgg19': models.vgg.VGG19,
        }[config.arch]
    except KeyError as ex:
        raise ValueError('unknown architecture', ex)
    return model_fn(num_classes=num_classes, norm=norm)

if __name__ == '__main__':
    app.run(main)
