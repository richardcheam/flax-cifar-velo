import collections
from functools import partial
import sys
from typing import Callable, Tuple

from absl import app
from absl import flags
from flax import linen as nn
import jax
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

import models.densenet
import models.resnet_v1
import models.resnet_v2
import models.vgg
import models.wide_resnet
import util

####################################################################################
import sys
sys.path.append('../')

import functools
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from jaxopt._src import base, tree_util
from jaxopt import OptaxSolver
import optax
from VeLO_training.config.optimizer import get_velo_optimizer #VeLO

class OptaxState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  internal_state: NamedTuple
  aux: Any

# we need to reimplement optax's OptaxSolver's lopt_update method to properly pass in the loss data that VeLO expects.
def lopt_update(self,
            params: Any,
            state: NamedTuple,
            *args,
            **kwargs) -> base.OptStep:
  """Performs one iteration of the optax solver.

  Args:
    params: pytree containing the parameters.
    state: named tuple containing the solver state.
    *args: additional positional arguments to be passed to ``fun``.
    **kwargs: additional keyword arguments to be passed to ``fun``.
  Returns:
    (params, state)
  """
  if self.pre_update:
    params, state = self.pre_update(params, state, *args, **kwargs)

  (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)

  # note the only difference between this function and the baseline 
  # optax.OptaxSolver.lopt_update is that `extra_args` is now passed.
  # if you would like to use a different optimizer, you will likely need to
  # remove these extra_args.

  # delta is like the -learning_rate * grad, though more complex in VeLO (detail in paper)
  delta, opt_state = self.opt.update(
    grad, state.internal_state, params, extra_args={"loss": value}
  )
  # applies the actual update to parameters
  params = self._apply_updates(params, delta)

  # Computes optimality error before update to re-use grad evaluation.
  new_state = OptaxState(iter_num=state.iter_num + 1,
                          error=tree_util.tree_l2_norm(grad),
                          value=value,
                          aux=aux,
                          internal_state=opt_state)
  # return both updated params and training state used in the next iteration of training
  return base.OptStep(params=params, state=new_state)
####################################################################################


flags.DEFINE_string('dataset_root', None, 'Path to data.', required=True)
flags.DEFINE_bool('download', False, 'Download dataset.')
flags.DEFINE_integer('eval_batch_size', 128, 'Batch size to use during evaluation.')
flags.DEFINE_integer('loader_num_workers', 4, 'num_workers for DataLoader')
flags.DEFINE_integer('loader_prefetch_factor', 2, 'prefetch_factor for DataLoader')
flags.DEFINE_string('optimizer', None, 'Optimizer name', required=True)
config_flags.DEFINE_config_file('config')

FLAGS = flags.FLAGS
 
Dataset = torch.utils.data.Dataset
ModuleDef = Callable[..., nn.Module]

def main(_):
    config = ml_collections.ConfigDict(FLAGS.config)
 
    # Data loaders
    num_classes, input_shape, train_dataset, val_dataset = setup_data()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True,
        num_workers=FLAGS.loader_num_workers, prefetch_factor=FLAGS.loader_prefetch_factor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=FLAGS.eval_batch_size, shuffle=False,
        num_workers=FLAGS.loader_num_workers, prefetch_factor=FLAGS.loader_prefetch_factor)

    # Model init
    norm_kwargs = lambda t: {'use_running_average': not t}
    model = make_model(config, num_classes, input_shape, norm=nn.BatchNorm)
    rng = random.PRNGKey(0)
    init_vars = model.init(rng, jnp.zeros((1,)+input_shape), norm_kwargs=norm_kwargs(True))
    params, batch_stats = init_vars['params'], init_vars['batch_stats']

    def filter_kernel_params(tree):
        return [x for path, x in util.dict_tree_items(tree) if path[-1] == 'kernel']
    
    def loss_fun(params, data, mutable_vars):
        inputs, labels = data
        model_vars = {'params': params, **mutable_vars}
        logits, mutated_vars = model.apply(model_vars, inputs, norm_kwargs=norm_kwargs(True), mutable=['batch_stats'])

        # Use integer labels for stable loss
        #labels_int = labels_int.argmax(axis=-1).astype(jnp.int32)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))

        wd_vars = list(tree_util.tree_leaves(params)) if config.train.weight_decay_vars == 'all' else filter_kernel_params(params)
        wd_loss = 0.5 * sum(jnp.sum(jnp.square(x)) for x in wd_vars)
        loss_val = loss + config.train.weight_decay * wd_loss

        return loss_val, mutated_vars['batch_stats']

    # Setup OptaxSolver for any optimizer
    total_steps = config.train.num_epochs * len(train_loader)
    # base optimizer selection
    if FLAGS.optimizer == 'velo':
        base_opt = get_velo_optimizer(total_steps)
    elif FLAGS.optimizer == 'sgd':
        schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
        base_opt = optax.sgd(schedule)
    elif FLAGS.optimizer == 'sgdm':
        schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
        base_opt = optax.sgd(schedule, momentum=config.train.momentum)
    elif FLAGS.optimizer == 'adam':
        schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
        base_opt = optax.adam(schedule)
    elif FLAGS.optimizer == 'adamw':
        schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
        base_opt = optax.adamw(schedule, weight_decay=config.train.adam_weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {FLAGS.optimizer}")

    # wrap in OptaxSolver
    solver = OptaxSolver(
        opt=base_opt,
        fun=jax.value_and_grad(loss_fun, has_aux=True),
        maxiter=total_steps,
        has_aux=True,
        value_and_grad=True,
    )
    import types
    # Step 2: Attach lopt_update as a bound method
    solver.lopt_update = types.MethodType(lopt_update, solver)

    # initialize solver state using a sample batch
    # sample_inputs, sample_labels = next(iter(train_loader)) #one batch of data
    #convert to array change shape from (128, 3, 32, 32) to (128, 32, 32, 3)
    # x0 = jnp.moveaxis(jnp.asarray(sample_inputs.numpy()), -3, -1) 
    # # convert to jnp
    # y0_int = jnp.asarray(sample_labels.numpy())
    # # one-hot encode
    # y0 = jax.nn.one_hot(y0_int, num_classes)

    # grab one TRAIN batch to init state
    batch0 = next(iter(train_loader))
    # move PyTorch (N,C,H,W) → JAX (N,H,W,C)
    x0 = jnp.moveaxis(jnp.asarray(batch0[0].numpy()), -3, -1)
    y0 = jax.nn.one_hot(jnp.asarray(batch0[1].numpy()), num_classes)

    state = solver.init_state(params, (x0, y0), batch_stats)

    # choose update fn: custom for velo, default for others
    jitted_update = jax.jit(functools.partial(lopt_update, self=solver)) if FLAGS.optimizer == 'velo' else jax.jit(solver.update)
    
    # Eval function
    @jax.jit
    def eval_step(params, batch_stats, x):
        return model.apply({'params': params, 'batch_stats': batch_stats}, x, norm_kwargs=norm_kwargs(False))

    # Training & Validation loop
    for epoch in range(config.train.num_epochs + 1):
        train_accs, train_losses = [], []
        if epoch > 0:
            for inputs, labels in tqdm.tqdm(train_loader, desc=f"train ep {epoch}"):

                # change shape first because it is PyTorch otherwise shape error with JAX
                x = jnp.moveaxis(jnp.asarray(inputs.numpy()), -3, -1)
                y_int = jnp.asarray(labels.numpy())
                y = jax.nn.one_hot(y_int, num_classes)
                # update
                params, state = jitted_update(params, state, (x, y), batch_stats)
                batch_stats = state.aux

                logits = eval_step(params, batch_stats, x)
                loss = optax.softmax_cross_entropy(logits, y).mean().item()
                acc = (jnp.argmax(logits, axis=-1) == y).mean().item()

                train_accs.append(acc)
                train_losses.append(loss)

        # Validate at epoch 0 first 
        # Validation
        val_accs, val_losses = [], []
        for inputs, labels in tqdm.tqdm(val_loader, desc=f"val ep {epoch}"):
            x = jnp.moveaxis(jnp.asarray(inputs.numpy()), -3, -1)
            y_int = jnp.asarray(labels.numpy())
            y = jax.nn.one_hot(y_int, num_classes)

            logits = eval_step(params, batch_stats, x)
            loss = optax.softmax_cross_entropy(logits, y).mean().item()
            acc = (jnp.argmax(logits, axis=-1) == y).mean().item()

            val_losses.append(loss)
            val_accs.append(acc)

        train_acc = float(np.mean(train_accs))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_acc  = float(np.mean(val_accs))

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")



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


def make_model(
        config: ml_collections.ConfigDict,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        norm: ModuleDef = nn.BatchNorm) -> nn.Module:
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
    return  model_fn(num_classes=num_classes, norm=norm)


if __name__ == '__main__':
    app.run(main)
