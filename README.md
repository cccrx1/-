# CIFAR-10 Attack + REFINE Suite

This mini project integrates the following workflow into one entry script:

1. Train benign ResNet-18 on CIFAR-10.
2. Train three backdoor attacks on CIFAR-10:
   - BadNets
   - Blended
   - Label-Consistent
3. Train and evaluate REFINE defense for each attacked model.
4. Save structured metrics to JSON.

## File

- `run_cifar10_attack_refine_suite.py`: unified experiment entry.

## Environment Notes

Target runtime (as requested):

- PyTorch 2.2.0
- Python 3.10 (Ubuntu 22.04)
- MUSA 3.1.0 / 1x GPU

The script follows the original project style and keeps `core` interfaces unchanged.
If your environment uses CUDA-compatible APIs in PyTorch (common with many wrappers),
use `--device-mode GPU --cuda-selected-devices 0`.

## Dependencies

This project depends on:

- numpy
- Pillow
- opencv-python
- tqdm
- torch==2.2.0
- torchvision==0.17.0

Install with:

```bash
pip install -r requirements.txt
```

If you use a custom CUDA/MUSA build, install your platform-specific `torch` and `torchvision` first,
then run the command above.

## Localized Modules

This project now vendors the required `core` modules locally under:

- `projects/cifar10_attack_refine_suite/core/attacks`
- `projects/cifar10_attack_refine_suite/core/defenses`
- `projects/cifar10_attack_refine_suite/core/models`
- `projects/cifar10_attack_refine_suite/core/utils`

So the experiment entry script does not rely on the repository root `core` package anymore.

## Main Outputs

- Per-stage training logs from core modules under:
  - `projects/cifar10_attack_refine_suite/runs/...`
- Unified pipeline log:
  - `projects/cifar10_attack_refine_suite/runs/pipeline.log`
- Unified metrics summary:
  - `projects/cifar10_attack_refine_suite/runs/metrics_summary.json`

## Example Command

Run from repository root:

```bash
python projects/cifar10_attack_refine_suite/run_cifar10_attack_refine_suite.py \
  --dataset-root ./datasets \
  --output-root ./projects/cifar10_attack_refine_suite/runs \
  --adv-dataset-root ./projects/cifar10_attack_refine_suite/adv_dataset \
  --device-mode GPU \
  --cuda-selected-devices 0 \
  --batch-size 128 \
  --num-workers 8 \
  --y-target 0
```

## Default Parameters

Defaults are already tuned for a single high-memory GPU:

- Batch size: `128`
- Workers: `8`
- Benign epochs: `200`
- Attack epochs: `200`
- REFINE epochs: `150`
- Poisoned rate: BadNets `0.05`, Blended `0.05`, LC `0.25`
- LC PGD: `eps=8.0, alpha=1.5, steps=100`
