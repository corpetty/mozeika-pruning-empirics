import sys
sys.path.insert(0, ".")
from vgg16_pruning_v2 import VGGPruningConfig, run_pruning_experiment

cfg = VGGPruningConfig(
    device="cuda:1",
    seed=42,
    batch_size=64,
    image_size=224,
    fisher_batches=3,
    max_pruning_rounds=50,
    target_global_sparsity=0.99,
    prune_fraction_cap=0.15,
    lr=1e-4,
    train_epochs_per_round=1,
    finetune_epochs=1,
    data_root="/home/petty/.openclaw/workspace-ai-research/data",
)

run_pruning_experiment(cfg)
