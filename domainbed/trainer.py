import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from domainbed.quan.utils import find_modules_to_quantize, replace_module_by_names
from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(args_q, test_envs, args, hparams, q_steps, quant, checkpoint_freq, logger, writer, target_env=None):
    logger.info("Starting training...")

    args.real_test_envs = test_envs  # For logging
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    # Fix KeyError for 'indomain_test'
    if hparams.get("training", {}).get("indomain_test", 0.0) > 0.0:
        logger.info("!!! In-domain test mode On !!!")
        assert hparams.get("val_augment", False) is False, (
            "indomain_test splits the val set into val/test sets. "
            "Therefore, the val set should not be augmented."
        )

        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(args.trial_seed, env_i)
            val_split, test_split = split_dataset(out_split, n, seed=seed)
            val_splits.append((val_split, None))
            test_splits.append((test_split, None))

            logger.info(f"env {env_i}: out (# {len(out_split)}) -> val (# {len(val_split)}) / test (# {len(test_split)})")

        out_splits = val_splits  # Updated splits

    n_envs = len(dataset)
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=int)

    # Ensure batch_sizes can handle list-based test_envs
    for test_env in test_envs:
        batch_sizes[test_env] = 0

    batch_sizes = batch_sizes.tolist()

    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
        if batch_size > 0
    ]
    steps_per_epoch = min(steps_per_epochs) if steps_per_epochs else 1

    # Set the desired number of epochs
    desired_epochs = 50
    n_steps = int(desired_epochs * steps_per_epoch)

    logger.info(f"Total epochs set: {desired_epochs}, Total steps: {n_steps}")

    train_loaders = [
        InfiniteDataLoader(
            dataset=hparams["dataset"],
            weights=torch.ones(len(env)),  # Fix weight issue
            batch_size=hparams["batch_size"],
            num_workers=2,  # Adjust if slow
        )
        for (env, env_weights) in iterator.train(zip(in_splits, batch_sizes))
    ]

    # Initialize model before quantization
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    ).to(device)  # Ensure model is moved to device immediately

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(list)
    records = []
    epochs_path = Path(args.out_dir) / "results.jsonl"

    for step in range(n_steps):
        epoch = step // steps_per_epoch
        if step % steps_per_epoch == 0:
            logger.info(f"Starting epoch {epoch + 1}/{desired_epochs}")

        # Handle quantization at specific step
        if step == q_steps and quant == 1:
            modules_to_replace = find_modules_to_quantize(algorithm, args_q.quan)
            algorithm = replace_module_by_names(algorithm, modules_to_replace).to(device)  # Ensure new model is on device

        step_start_time = time.time()

        # Fetch next batch
        batches_dictlist = next(train_minibatches_iterator)
        batches = misc.merge_dictlist(batches_dictlist)

        # Ensure proper device handling
        for key, tensorlist in batches.items():
            if isinstance(tensorlist, list):
                batches[key] = [tensor.to(device) for tensor in tensorlist]
            else:
                batches[key] = tensorlist.to(device)

        # Forward and update step
        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)

        # Log step values
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        # Periodic checkpoint saving
        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": epoch,
                **{key: np.mean(val) for key, val in checkpoint_vals.items()},
            }

            logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {results.get('loss', 'N/A')}")

            checkpoint_vals = collections.defaultdict(list)

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=str) + "\n")

            writer.add_scalars_with_prefix(step_vals, step, "summary/")

    return records
