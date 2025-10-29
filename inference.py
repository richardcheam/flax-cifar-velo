import argparse
import os
from glob import glob
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import checkpoints
import torch
from torchvision import datasets, transforms
import models.wide_resnet  # adjust if needed

norm = nn.BatchNorm
norm_kwargs = lambda train: {'use_running_average': not train}

# ----- CLI Args -----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wrn28_2")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--opt", required=True)
    #p.add_argument("--arch", default="wrn28_2")
    p.add_argument("--ckpt_dir", default="results/checkpoints")
    return p.parse_args()

args = parse_args()

# ----- Dataset Loader -----
def setup_test_loader(dataset_root: str, batch_size: int):
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    test_dataset = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=transform_eval)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2
    )
    return test_loader, 10  # 10 classes

# ----- Model Setup -----
def build_model(model_name: str, num_classes: int):
    if model_name == "wrn28_2":
        return models.wide_resnet.WideResNet(depth=28, width=2, num_classes=num_classes)
    elif model_name == "wrn28_8":
        return models.wide_resnet.WideResNet(depth=28, width=8, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")

# ----- Evaluation -----
def evaluate(params, batch_stats, model, test_loader) -> Tuple[float, float]:
    accs = []
    losses = []

    for inputs, labels in test_loader:
        inputs = jnp.asarray(inputs.numpy())
        labels = jnp.asarray(labels.numpy())
        inputs = jnp.moveaxis(inputs, 1, -1)  # CHW → HWC

        logits = model.apply({'params': params, 'batch_stats': batch_stats}, inputs,
                norm_kwargs=norm_kwargs(train=False))
        logits = logits.astype(jnp.float32)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        acc = jnp.argmax(logits, axis=-1) == labels

        accs.append(np.array(acc))
        losses.append(np.array(loss))

    mean_acc = np.mean(np.concatenate(accs))
    mean_loss = np.mean(np.concatenate(losses))
    return mean_acc, mean_loss

# ----- Load All Checkpoints for Seeds -----
def get_ckpts(ckpt_dir, model, opt):
    pattern = os.path.join(ckpt_dir, f"cifar10/{opt}_{model}_cifar10_seed*_pretrained.msgpack")
    return sorted(glob(pattern))

# ----- Load Checkpoint -----
def restore_checkpoint(path: str, model, input_shape=(32, 32, 3)):
    dummy_input = jnp.zeros((1,) + input_shape, dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy_input, norm_kwargs={'use_running_average': True})
    restored = checkpoints.restore_checkpoint(path, target=None)
    params = restored['params']
    batch_stats = restored.get('batch_stats', variables.get('batch_stats', {}))
    return params, batch_stats

# ----- Main Evaluation -----
def main():
    test_loader, num_classes = setup_test_loader(args.dataset_root, args.batch_size)
    model = build_model(args.model, num_classes)
    ckpts = get_ckpts(args.ckpt_dir, args.model, args.opt)

    accs, losses = [], []

    for ckpt_path in ckpts:
        print(f"Evaluating {ckpt_path}")
        params, batch_stats = restore_checkpoint(ckpt_path, model)
        acc, loss = evaluate(params, batch_stats, model, test_loader)
        accs.append(acc)
        losses.append(loss)

    print(f"\nAggregated Test Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Aggregated Test Loss:     {np.mean(losses):.4f} ± {np.std(losses):.4f}")

if __name__ == "__main__":
    main()

