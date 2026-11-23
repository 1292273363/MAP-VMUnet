# Third-party notices

## VM-UNet and its backbone

[`map_vmunet/backbone.py`](map_vmunet/backbone.py) adapts `models/vmunet/vmamba.py` from [JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet), licensed under [Apache-2.0](LICENSE). The public version retains the patch embedding and resolution changes, VSS/SS2D blocks, and segmentation projection used by MAP-VMUnet. [`map_vmunet/model.py`](map_vmunet/model.py) extends the encoder and decoder flow for MDK and PARD. The training implementation also follows the VM-UNet training lineage.

Jiacheng Ruan, Jincheng Li, and Suncheng Xiang. *VM-UNet: Vision Mamba UNet for Medical Image Segmentation*. ACM Transactions on Multimedia Computing, Communications, and Applications (2025). [Published article](https://doi.org/10.1145/3767748) · [Preprint](https://arxiv.org/abs/2402.02491)

VM-UNet credits [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). The VSS and SS2D design originates with Yue Liu et al., *VMamba: Visual State Space Model* (NeurIPS 2024). The [VMamba MIT notice](licenses/VMamba-MIT.txt) accompanies the backbone. [VMamba paper](https://arxiv.org/abs/2401.10166)

The selective-scan operator is imported from the external [`mamba-ssm`](https://github.com/state-spaces/mamba) package under [Apache-2.0](https://github.com/state-spaces/mamba/blob/main/LICENSE). Credit Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [Mamba paper](https://arxiv.org/abs/2312.00752)

## MDK and PARD

The multi-kernel depthwise branches and channel shuffle in [`map_vmunet/mdk.py`](map_vmunet/mdk.py) draw on [SLDGroup/MK-UNet](https://github.com/SLDGroup/MK-UNet). MAP-VMUnet adapts those ideas for VM-UNet features with sample-dependent branch weighting and high-frequency calibration. The [MK-UNet BSD-3-Clause notice](licenses/MK-UNet-BSD-3-Clause.txt) accompanies this attribution.

Md Mostafijur Rahman and Radu Marculescu. *MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation*. ICCV Workshops (2025). [Official repository and citation](https://github.com/SLDGroup/MK-UNet)

The original MAP-VMUnet implementation credits [EccoMamba](https://doi.org/10.1609/aaai.v40i13.38109) for its encoder stage placement study and [CCViM](https://github.com/zymissy/CCViM) for its combination of global VSS features with local compensation.

The uncertainty feedback in [`map_vmunet/pard.py`](map_vmunet/pard.py) draws on Taehun Kim, Hyemin Lee, and Daijin Kim, *UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation* (ACM Multimedia 2021). [UACANet paper](https://arxiv.org/abs/2107.02368)

## Data and model weights

Dataset and checkpoint terms come from their respective providers. Use the [ISIC Challenge data page](https://challenge.isic-archive.com/data/) for ISIC 2017 and ISIC 2018 and the [official VM-UNet instructions](https://github.com/JCruan519/VM-UNet#2-prepare-the-pre_trained-weights) for the externally hosted VMamba initialization checkpoint.
