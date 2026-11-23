# MAP-VMUnet

MAP-VMUnet extends VM-UNet for skin lesion segmentation with a multi-scale dynamic kernel block (MDK) and a progressive ambiguity-guided residual decoder (PARD). The repository contains the complete model and the two single-module configurations in the paper's main ablation table.

| Configuration | MDK | PARD | Datasets used for the corresponding runs |
| --- | --- | --- | --- |
| `map_vmunet` | Yes | Yes | ISIC 2017 and ISIC 2018 |
| `mdk_only` | Yes | No | ISIC 2018 |
| `pard_only` | No | Yes | ISIC 2018 |

`train.py` is the shared training and evaluation entry point. `map_vmunet/` contains the model, data loader, and training code. `configs/` contains the three settings above, and `splits/` contains the fixed sample lists.

## Environment

Use Linux or WSL 2 with an NVIDIA GPU and a CUDA-enabled PyTorch build. The reference environment uses Python 3.8, PyTorch 1.13.1 with CUDA 11.7, torchvision 0.14.1, and Mamba 1.0.1.

```bash
conda create -n map-vmunet python=3.8
conda activate map-vmunet
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python -m pip install triton==2.0.0
python -m pip install \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.0.0/causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl'
python -m pip install \
  'https://github.com/state-spaces/mamba/releases/download/v1.0.1/mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl'
```

The commands use the upstream Linux x86-64 wheels for Python 3.8, PyTorch 1.13 and the default PyTorch C++ ABI. If building from source, use a compatible CUDA toolkit.

```bash
python train.py --config map_vmunet --smoke
```

The smoke test uses synthetic inputs and runs a forward pass, a backward pass, one optimizer step, and an evaluation pass without dataset files.

## Data and initialization

Obtain the images and segmentation masks under the [ISIC Challenge data terms](https://challenge.isic-archive.com/data/). The bundled split manifests refer to the prepared data pools used by the study. Use the pool described in [VM-UNet's ISIC preparation instructions](https://github.com/JCruan519/VM-UNet#1-prepare-the-dataset) and preserve its source folder and filename mapping, especially the numeric filenames for ISIC 2018. The ISIC 2018 lists use legacy identifiers and require that mapping when starting from official archives. For ISIC 2017, use the original ISIC image IDs and the filenames and extensions in the manifest.

```text
data/
  isic2017/
    train/{images,masks}/
    val/{images,masks}/
  isic2018/
    train/{images,masks}/
    val/{images,masks}/
```

The folders above identify the source files. The study's train, validation, and independent test membership comes from the lists under `splits/`. ISIC 2017 has 1,504 / 215 / 429 samples and ISIC 2018 has 1,886 / 269 / 539. These are fixed image-level splits generated with seed 42.

Set `--data-root` to the directory containing the two dataset folders. Download the VMamba initialization checkpoint `vmamba_small_e238_ema.pth` from the [VM-UNet pretrained weights instructions](https://github.com/JCruan519/VM-UNet#2-prepare-the-pre_trained-weights) and pass its location with `--pretrained`.

```bash
python train.py --dataset isic2018 --data-root ./data --check-data
```

## Train and evaluate

```bash
python train.py --dataset isic2018 --config map_vmunet \
  --pretrained ./weights/vmamba_small_e238_ema.pth --output ./runs/isic2018-map
python train.py --dataset isic2018 --config mdk_only \
  --pretrained ./weights/vmamba_small_e238_ema.pth --output ./runs/isic2018-mdk-only
python train.py --dataset isic2018 --config pard_only \
  --pretrained ./weights/vmamba_small_e238_ema.pth --output ./runs/isic2018-pard-only
```

For the ISIC 2017 main run, use `--dataset isic2017` with a new output directory. Set `--gpu` if needed. Each run records its configuration and validation history and selects `best_dice.pth` using the maximum per-image mean Dice on the validation split. Evaluation on the held-out test split saves `test_summary.json` and `test_case_metrics.csv`.

```bash
python train.py --dataset isic2018 --config map_vmunet \
  --evaluate ./runs/isic2018-map/best_dice.pth --output ./runs/isic2018-map-test
```

Inputs are resized to 256 × 256 with bilinear interpolation for images and masks. The train loader uses paired flips and a rotation angle drawn once at initialization. The released entry point records that angle for new runs. Historical logs lack the sampled angle, so exact numerical agreement with historical runs is not guaranteed. Main Dice and IoU are per-image means. Accuracy, sensitivity, and specificity use totals across the test split.

## Attribution and license

This code builds on Jiacheng Ruan, Jincheng Li, and Suncheng Xiang's [VM-UNet](https://github.com/JCruan519/VM-UNet), *VM-UNet: Vision Mamba UNet for Medical Image Segmentation*, ACM TOMM, [doi:10.1145/3767748](https://doi.org/10.1145/3767748). See [third-party notices](THIRD_PARTY_NOTICES.md) for the reused code, dependencies, and upstream licenses. The repository code is provided under [Apache-2.0](LICENSE). Data and externally hosted checkpoints have their own terms.
