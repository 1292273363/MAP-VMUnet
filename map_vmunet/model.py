"""MAP-VMUnet and its two single-component configurations.

The encoder and decoder build on VM-UNet (Apache-2.0). See
THIRD_PARTY_NOTICES.md for upstream sources and the implementation lineage.
"""

from collections import OrderedDict
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import VSSM
from .mdk import DynamicMultiKernelBlock
from .pard import ProgressiveAmbiguityDecoder


def _validate_stages(stages: Iterable[int], num_stages: int, name: str):
    stages = tuple(sorted(set(int(stage) for stage in stages)))
    invalid = [stage for stage in stages if stage < 0 or stage >= num_stages]
    if invalid:
        raise ValueError(
            "{} contains invalid stages {}; valid range is [0, {})".format(
                name, invalid, num_stages
            )
        )
    return stages


class MAPVMUNet(nn.Module):
    """VM-UNet with independently configurable MDK and PARD components."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        depths: Sequence[int] = (2, 2, 9, 2),
        depths_decoder: Sequence[int] = (2, 9, 2, 2),
        drop_path_rate: float = 0.2,
        load_ckpt_path: Optional[str] = None,
        use_mdk: bool = True,
        use_pard: bool = True,
        mdk_stages: Sequence[int] = (0, 1, 2),
        mdk_kernels: Sequence[int] = (1, 3, 5),
        use_dilated_kernel: bool = False,
        dilated_kernel_dilation: int = 2,
        mdk_expansion: int = 2,
        pard_stages: Sequence[int] = (0, 1, 2, 3),
        pard_feature_hidden_ratio: float = 0.25,
        pard_prediction_hidden_ratio: float = 0.5,
        pard_feature_epsilon: float = 0.1,
        pard_prediction_epsilon: float = 0.1,
        pard_alpha_init: float = 1e-3,
        pard_alpha_bound: float = 1.0,
        pard_delta_max: float = 2.0,
        pard_eta_init: float = 0.1,
        pard_detach_ambiguity: bool = True,
        pard_ambiguity_temperature: float = 1.0,
        init_gamma: float = 1e-3,
    ):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.use_mdk = bool(use_mdk)
        self.use_pard = bool(use_pard)

        self.vmunet = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            drop_path_rate=drop_path_rate,
        )
        self.mdk_stages = _validate_stages(
            mdk_stages, self.vmunet.num_layers, "mdk_stages"
        )
        self.pard_stages = _validate_stages(
            pard_stages, self.vmunet.num_layers, "pard_stages"
        )
        self.mdk_blocks = nn.ModuleDict()
        if self.use_mdk:
            for stage in self.mdk_stages:
                self.mdk_blocks[str(stage)] = DynamicMultiKernelBlock(
                    channels=self.vmunet.dims[stage],
                    expansion=mdk_expansion,
                    kernels=mdk_kernels,
                    use_dilated_kernel=use_dilated_kernel,
                    dilated_kernel_dilation=dilated_kernel_dilation,
                    shuffle_groups=min(4, self.vmunet.dims[stage]),
                    init_gamma=init_gamma,
                )
        self.pard = None
        if self.use_pard:
            if self.num_classes != 1:
                raise ValueError("PARD currently supports binary segmentation only")
            self.pard = ProgressiveAmbiguityDecoder(
                stage_channels=self.vmunet.dims,
                final_channels=self.vmunet.final_conv.in_channels,
                stages=self.pard_stages,
                feature_hidden_ratio=pard_feature_hidden_ratio,
                prediction_hidden_ratio=pard_prediction_hidden_ratio,
                feature_epsilon=pard_feature_epsilon,
                prediction_epsilon=pard_prediction_epsilon,
                alpha_init=pard_alpha_init,
                alpha_bound=pard_alpha_bound,
                delta_max=pard_delta_max,
                eta_init=pard_eta_init,
                detach_ambiguity=pard_detach_ambiguity,
                ambiguity_temperature=pard_ambiguity_temperature,
            )

    def forward_features(self, x: torch.Tensor, return_global: bool = False):
        """Insert MDK before VSS and optionally expose post-VSS features."""
        skip_list = []
        global_list = []
        x = self.vmunet.patch_embed(x)
        x = self.vmunet.pos_drop(x)

        for stage, layer in enumerate(self.vmunet.layers):
            if str(stage) in self.mdk_blocks:
                x = self.mdk_blocks[str(stage)](x)
            skip_list.append(x)
            if return_global:
                global_feature = layer.forward_blocks(x)
                x = (
                    layer.downsample(global_feature)
                    if layer.downsample is not None
                    else global_feature
                )
                global_list.append(global_feature)
            else:
                x = layer(x)
        if return_global:
            return x, skip_list, global_list
        return x, skip_list

    def _forward_pard(
        self,
        x: torch.Tensor,
        local_features,
        global_features,
    ):
        """Run coarse-to-fine ambiguity-guided feature and logit correction."""
        x = self.vmunet.layers_up[0](x)
        logits, ambiguity = self.pard.initialize(x)
        auxiliary_logits = [logits]

        # Stages 3->2->1 use the same MDK skips as the existing decoder.  PARD
        # adds only a bounded ambiguity-gated residual around that base path.
        for decoder_index in range(1, self.vmunet.num_layers):
            encoder_stage = self.vmunet.num_layers - decoder_index
            x = self.pard.fuse(
                stage=encoder_stage,
                decoder=x,
                local=local_features[encoder_stage],
                global_feature=global_features[encoder_stage],
                ambiguity=ambiguity,
                add_local_skip=True,
            )
            x = self.vmunet.layers_up[decoder_index](x)
            target_stage = encoder_stage - 1
            logits, ambiguity = self.pard.refine_stage(target_stage, x, logits)
            auxiliary_logits.append(logits)

        # At stage zero the local feature enters the PARD correction only.
        x = self.pard.fuse(
            stage=0,
            decoder=x,
            local=local_features[0],
            global_feature=global_features[0],
            ambiguity=ambiguity,
            add_local_skip=False,
        )
        final_feature = self.vmunet.final_up(x)
        base_logits = self.vmunet.final_conv(
            final_feature.permute(0, 3, 1, 2).contiguous()
        )
        logits, ambiguity = self.pard.refine_final(final_feature, logits, base_logits)

        return {
            "seg": torch.sigmoid(logits),
            "seg_logits": logits,
            "aux_seg": [torch.sigmoid(item) for item in auxiliary_logits],
            "aux_logits": auxiliary_logits,
            "ambiguity": ambiguity,
        }

    def forward(self, x: torch.Tensor):
        """Return the segmentation or, with PARD, its intermediate predictions."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.pard is not None:
            x, skip_list, global_list = self.forward_features(x, return_global=True)
            return self._forward_pard(x, skip_list, global_list)

        x, skip_list = self.forward_features(x)
        x = self.vmunet.forward_features_up(x, skip_list)
        logits = self.vmunet.forward_final(x)
        return torch.sigmoid(logits) if self.num_classes == 1 else logits

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint

    @staticmethod
    def _strip_known_prefixes(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("vmunet."):
            key = key[len("vmunet.") :]
        return key

    def _remap_pretrained(self, state_dict):
        """Map VMamba/VM-UNet keys into this wrapper, including decoder mirrors."""
        remapped = OrderedDict()
        for key, value in state_dict.items():
            backbone_key = self._strip_known_prefixes(key)
            remapped["vmunet." + backbone_key] = value

            for encoder_stage, decoder_stage in ((0, 3), (1, 2), (2, 1), (3, 0)):
                marker = "layers.{}.".format(encoder_stage)
                if marker in backbone_key:
                    decoder_key = backbone_key.replace(
                        marker, "layers_up.{}.".format(decoder_stage), 1
                    )
                    remapped["vmunet." + decoder_key] = value
                    break
        return remapped

    def load_from(self, checkpoint_path: Optional[str] = None):
        """Load VMamba/VM-UNet weights with shape filtering and ``strict=False``."""
        checkpoint_path = checkpoint_path or self.load_ckpt_path
        if checkpoint_path is None:
            print("No pretrained checkpoint configured; skipping load.")
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source = self._extract_state_dict(checkpoint)
        remapped = self._remap_pretrained(source)
        target = self.state_dict()

        compatible = OrderedDict()
        unexpected_keys = []
        shape_mismatches = []
        for key, value in remapped.items():
            if key not in target:
                unexpected_keys.append(key)
            elif target[key].shape != value.shape:
                shape_mismatches.append(
                    "{}: checkpoint {} != model {}".format(
                        key, tuple(value.shape), tuple(target[key].shape)
                    )
                )
            else:
                compatible[key] = value

        if not compatible:
            raise ValueError(
                "The checkpoint has no tensors compatible with the VM-UNet backbone."
            )
        incompatible = self.load_state_dict(compatible, strict=False)
        combined_unexpected = list(incompatible.unexpected_keys) + unexpected_keys
        print(
            "Loaded {} / {} remapped tensors from {}".format(
                len(compatible), len(remapped), checkpoint_path
            )
        )
        print("missing_keys:", list(incompatible.missing_keys))
        print("unexpected_keys:", combined_unexpected)
        print("shape_mismatches:", shape_mismatches)
        return incompatible
