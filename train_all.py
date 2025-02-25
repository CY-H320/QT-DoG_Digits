import argparse
import collections
import random
import sys
from pathlib import Path
import yaml
import munch
import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train
from domainbed.networks.resnet8 import ResNet8

def parse_arguments():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*", default=[])
    parser.add_argument("--data_dir", type=str, default="data/Digits")
    parser.add_argument("--dataset", type=str, default="Digits")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--quant", type=int, default=0)
    parser.add_argument("--trial_seed", type=int, default=0, help="Trial seed")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--q_steps", type=int, default=2000, help="Quantization steps")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Checkpoint frequency")
    parser.add_argument("--test_envs", type=int, nargs="+", default=None, help="Test environments")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", type=int, default=None, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", type=int, default=10)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--show", action="store_true", help="Show configs and exit")
    parser.add_argument("--evalmode", default="fast", help="Evaluation mode: [fast, all]")
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    
    return parser.parse_known_args()

def setup_logger(args):
    args.out_root = Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    return writer, logger

def main():
    args, left_argv = parse_arguments()
    
    with open("config_digits.yaml") as f:
        cfg = munch.munchify(yaml.safe_load(f))

    
    hparams = Config("config_digits.yaml", *args.configs, default=hparams_registry.default_hparams(args.algorithm, args.dataset))
    hparams.argv_update(left_argv)
    
    if args.debug:
        args.checkpoint_freq, args.steps = 5, 10
        args.name += "_debug"
    
    args.unique_name = f"{misc.timestamp()}_{args.name}"
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)
    
    writer, logger = setup_logger(args)
    
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    
    dataset, _, _ = get_dataset([0], args, hparams)
    logger.info(f"Dataset: {args.dataset}, Environments: {len(dataset)}, Classes: {dataset.num_classes}")
    
    n_steps = (args.steps or dataset.N_STEPS) // (args.checkpoint_freq or dataset.CHECKPOINT_FREQ) * (args.checkpoint_freq or dataset.CHECKPOINT_FREQ) + 1
    logger.info(f"Adjusted steps: {n_steps}")
    
    args.test_envs = args.test_envs or [[te] for te in range(len(dataset))]
    logger.info(f"Test environments: {args.test_envs}")
    
    all_records, results = [], collections.defaultdict(list)
    for test_env in args.test_envs:
        res, records = train(cfg, test_env, args=args, hparams=hparams, q_steps=args.q_steps, quant=args.quant, checkpoint_freq=args.checkpoint_freq, logger=logger, writer=writer)
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)
    
    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    for key, row in results.items():
        table.add_row([key] + [f"{acc:.3%}" for acc in row + [np.mean(row)]])
    logger.info(table)

if __name__ == "__main__":
    main()
