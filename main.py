import argparse
from omegaconf import OmegaConf
from lib.Trainer import Trainer
import torch
import warnings
# Suppress specific warnings - it doesnt matter in inference
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning)

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 128
cfg = OmegaConf.merge(OmegaConf.load('config/config.yaml'), OmegaConf.load('config/path.yaml'))

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Root directory of datasets.')
parser.add_argument('--grid_res', type=int)
parser.add_argument('--shapenet_id', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--ga', type=int)
args = parser.parse_args()

if args.name is not None:
    OmegaConf.update(cfg, 'name', args.name)
    OmegaConf.update(cfg, 'results_folder', args.name)

if args.batch_size is not None:
    OmegaConf.update(cfg, 'training.batch_size', args.batch_size)

if args.ga is not None:
    OmegaConf.update(cfg, 'training.ga', args.ga)

if args.shapenet_id is not None:
    OmegaConf.update(cfg, 'dataset.shapenet_ids', [args.shapenet_id])

if args.data_path is not None:
    OmegaConf.update(cfg, 'data_path', args.data_path)

if args.grid_res is not None:
    OmegaConf.update(cfg, 'dataset.grid_res', args.grid_res)


print(cfg)

import os
os.makedirs(cfg.results_folder,exist_ok=True)

with open(cfg.results_folder+"/config.yaml", "w") as f:
   OmegaConf.save(cfg, f)

trainer = Trainer(
    cfg=cfg,
    train_batch_size = cfg.training.batch_size,
    save_and_sample_every = cfg.training.test_every,
    results_folder = cfg.results_folder,
    config_folder=  cfg.results_folder,
    num_samples = 1,
    train_lr = cfg.training.lr,
    train_num_steps = cfg.training.num_steps,         # total training steps
    gradient_accumulate_every = cfg.training.ga,    # gradient accumulation steps
    ema_decay = cfg.training.ema_decay,                # exponential moving average decay
)

#if cfg.load_weights:
#    trainer.load(cfg.num_weights)
trainer.train()
