"""Progressive ambiguity-guided residual decoder for MAP-VMUnet.

The module follows a coarse-to-fine prediction-refinement pattern: the current
logit map defines a continuous ambiguity map, the ambiguity map gates a
feature-correction residual, and a bounded prediction head updates the logits
at the next finer scale.  It is an original VM-UNet adaptation informed by the
progressive uncertainty feedback of UACANet and the residual prediction update
of reverse-attention decoders.

All encoder/decoder features use VM-UNet's native BHWC layout.  Prediction and
ambiguity maps use B1HW.
"""

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, requested: int = 8) -> int:
    """Return a small valid GroupNorm group count."""
    return max(1, math.gcd(int(channels), min(int(channels), int(requested))))


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _atanh_ratio(value: float, bound: float) -> float:
    if bound <= 0:
        raise ValueError("bound must be positive")
    ratio = min(max(float(value) / float(bound), -0.999999), 0.999999)
    return 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))


def ambiguity_from_logits(
    logits: torch.Tensor,
    detach: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return smooth Bernoulli ambiguity ``4*p*(1-p)`` in ``[0, 1]``.

    ``detach=True`` stops the routing signal from encouraging the predictor to
    become uncertain merely to open a correction gate.  The prediction logits
    themselves remain on the main differentiable residual path.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    source = logits.detach() if detach else logits
    probability = torch.sigmoid(source / float(temperature))
    return (4.0 * probability * (1.0 - probability)).clamp_(0.0, 1.0)


class HighFrequencyExtractor(nn.Module):
    """Parameter-free local residual ``x - AvgPool3x3(x)`` for BCHW features."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected BCHW tensor, got {}".format(tuple(x.shape)))
        smooth = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        )
        return x - smooth


class CoarsePredictionHead(nn.Module):
    """Predict the initial coarse B1HW logit map from a BHWC feature map."""

    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(16, int(round(int(channels) * float(hidden_ratio))))
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[-1] != self.channels:
            raise ValueError(
                "expected BHWC with C={}, got {}".format(
                    self.channels, tuple(feature.shape)
                )
            )
        return self.net(feature).permute(0, 3, 1, 2).contiguous()


class BoundedResidualHead(nn.Module):
    """Predict a bounded logit correction from a BHWC feature map."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.5,
        delta_max: float = 2.0,
    ):
        super().__init__()
        if delta_max <= 0:
            raise ValueError("delta_max must be positive")
        self.channels = int(channels)
        self.delta_max = float(delta_max)
        hidden = max(16, int(round(self.channels * float(hidden_ratio))))
        self.norm = nn.LayerNorm(self.channels)
        self.fc1 = nn.Linear(self.channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature: torch.Tensor):
        if feature.ndim != 4 or feature.shape[-1] != self.channels:
            raise ValueError(
                "expected BHWC with C={}, got {}".format(
                    self.channels, tuple(feature.shape)
                )
            )
        raw = self.fc2(self.act(self.fc1(self.norm(feature))))
        raw = raw.permute(0, 3, 1, 2).contiguous()
        return self.delta_max * torch.tanh(raw)


class AmbiguityGuidedFusion(nn.Module):
    """Gate a learned correction while preserving the existing MDK skip path.

    For regular VM-UNet skip stages the base is ``decoder + local``.  At stage
    zero, which the official decoder does not directly consume, ``add_local_skip``
    is false and the base is the decoder feature alone.
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.25,
        epsilon: float = 0.1,
        alpha_init: float = 1e-3,
        alpha_bound: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if abs(alpha_init) >= alpha_bound:
            raise ValueError("abs(alpha_init) must be smaller than alpha_bound")
        self.channels = int(channels)
        self.epsilon = float(epsilon)
        self.alpha_bound = float(alpha_bound)
        hidden = max(16, int(round(self.channels * float(hidden_ratio))))

        self.high_frequency = HighFrequencyExtractor(kernel_size=3)
        self.decoder_proj = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.global_proj = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.high_proj = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, kernel_size=1, bias=True),
        )
        self.raw_alpha = nn.Parameter(
            torch.tensor(_atanh_ratio(alpha_init, self.alpha_bound))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_bound * torch.tanh(self.raw_alpha)

    def forward(
        self,
        decoder: torch.Tensor,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        ambiguity: torch.Tensor,
        add_local_skip: bool = True,
    ) -> torch.Tensor:
        tensors = (decoder, local, global_feature)
        if any(x.ndim != 4 or x.shape[-1] != self.channels for x in tensors):
            raise ValueError(
                "decoder/local/global must be BHWC with C={}".format(self.channels)
            )
        if decoder.shape != local.shape or decoder.shape != global_feature.shape:
            raise ValueError(
                "feature shapes must match: {}, {}, {}".format(
                    tuple(decoder.shape),
                    tuple(local.shape),
                    tuple(global_feature.shape),
                )
            )

        decoder_cf = decoder.permute(0, 3, 1, 2).contiguous()
        local_cf = local.permute(0, 3, 1, 2).contiguous()
        global_cf = global_feature.permute(0, 3, 1, 2).contiguous()
        high_cf = self.high_frequency(local_cf)
        residual = self.mix(
            torch.cat(
                [
                    self.decoder_proj(decoder_cf),
                    self.global_proj(global_cf),
                    self.high_proj(high_cf),
                ],
                dim=1,
            )
        )

        if ambiguity.ndim != 4 or ambiguity.shape[1] != 1:
            raise ValueError("ambiguity must be B1HW")
        if ambiguity.shape[-2:] != residual.shape[-2:]:
            ambiguity = F.interpolate(
                ambiguity,
                size=residual.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = self.epsilon + (1.0 - self.epsilon) * ambiguity
        base = decoder_cf + local_cf if add_local_skip else decoder_cf
        output = base + self.alpha * gate * residual
        return output.permute(0, 2, 3, 1).contiguous()


class ProgressiveLogitRefiner(nn.Module):
    """Upsample a logit map and add an ambiguity-gated bounded correction."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.5,
        delta_max: float = 2.0,
        eta_init: float = 0.1,
        epsilon: float = 0.1,
        detach_ambiguity: bool = True,
        ambiguity_temperature: float = 1.0,
    ):
        super().__init__()
        if not 0.0 < eta_init < 1.0:
            raise ValueError("eta_init must be in (0, 1)")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)
        self.detach_ambiguity = bool(detach_ambiguity)
        self.ambiguity_temperature = float(ambiguity_temperature)
        self.residual_head = BoundedResidualHead(
            channels=channels,
            hidden_ratio=hidden_ratio,
            delta_max=delta_max,
        )
        self.raw_eta = nn.Parameter(torch.tensor(_logit(eta_init)))

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    def forward(
        self,
        feature: torch.Tensor,
        previous_logits: torch.Tensor,
        base_logits: torch.Tensor = None,
    ):
        delta = self.residual_head(feature)
        previous_ambiguity = ambiguity_from_logits(
            previous_logits,
            detach=self.detach_ambiguity,
            temperature=self.ambiguity_temperature,
        )
        resized_logits = F.interpolate(
            previous_logits,
            size=delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        resized_ambiguity = F.interpolate(
            previous_ambiguity,
            size=delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.epsilon + (1.0 - self.epsilon) * resized_ambiguity
        if base_logits is None:
            anchor_logits = resized_logits
        else:
            if base_logits.shape != resized_logits.shape:
                raise ValueError(
                    "base_logits shape {} does not match target {}".format(
                        tuple(base_logits.shape), tuple(resized_logits.shape)
                    )
                )
            anchor_logits = base_logits
        logits = anchor_logits + self.eta * gate * delta
        ambiguity = ambiguity_from_logits(
            logits,
            detach=self.detach_ambiguity,
            temperature=self.ambiguity_temperature,
        )
        return logits, ambiguity


class ProgressiveAmbiguityDecoder(nn.Module):
    """Coarse predictions with independently optional feature and logit refinements."""

    def __init__(
        self,
        stage_channels: Sequence[int],
        final_channels: int,
        stages: Iterable[int] = (0, 1, 2, 3),
        feature_hidden_ratio: float = 0.25,
        prediction_hidden_ratio: float = 0.5,
        feature_epsilon: float = 0.1,
        prediction_epsilon: float = 0.1,
        alpha_init: float = 1e-3,
        alpha_bound: float = 1.0,
        delta_max: float = 2.0,
        eta_init: float = 0.1,
        detach_ambiguity: bool = True,
        ambiguity_temperature: float = 1.0,
        use_feature_refinement: bool = True,
        use_logit_refinement: bool = True,
    ):
        super().__init__()
        self.stage_channels = tuple(int(c) for c in stage_channels)
        if len(self.stage_channels) < 2:
            raise ValueError("at least two encoder stages are required")
        self.stages = tuple(sorted(set(int(stage) for stage in stages)))
        invalid = [s for s in self.stages if s < 0 or s >= len(self.stage_channels)]
        if invalid:
            raise ValueError("invalid PARD stages: {}".format(invalid))
        self.detach_ambiguity = bool(detach_ambiguity)
        self.ambiguity_temperature = float(ambiguity_temperature)

        deepest = len(self.stage_channels) - 1
        self.coarse_head = CoarsePredictionHead(
            self.stage_channels[deepest], hidden_ratio=prediction_hidden_ratio
        )
        self.fusions = nn.ModuleDict(
            {
                str(stage): AmbiguityGuidedFusion(
                    channels=self.stage_channels[stage],
                    hidden_ratio=feature_hidden_ratio,
                    epsilon=feature_epsilon,
                    alpha_init=alpha_init,
                    alpha_bound=alpha_bound,
                )
                for stage in self.stages
                if use_feature_refinement
            }
        )
        self.stage_refiners = nn.ModuleDict(
            {
                str(stage): ProgressiveLogitRefiner(
                    channels=self.stage_channels[stage],
                    hidden_ratio=prediction_hidden_ratio,
                    delta_max=delta_max,
                    eta_init=eta_init,
                    epsilon=prediction_epsilon,
                    detach_ambiguity=detach_ambiguity,
                    ambiguity_temperature=ambiguity_temperature,
                )
                for stage in range(deepest)
                if use_logit_refinement
            }
        )
        self.final_refiner = None
        if use_logit_refinement:
            self.final_refiner = ProgressiveLogitRefiner(
                channels=int(final_channels),
                hidden_ratio=prediction_hidden_ratio,
                delta_max=delta_max,
                eta_init=eta_init,
                epsilon=prediction_epsilon,
                detach_ambiguity=detach_ambiguity,
                ambiguity_temperature=ambiguity_temperature,
            )

    def initialize(self, feature: torch.Tensor):
        logits = self.coarse_head(feature)
        ambiguity = ambiguity_from_logits(
            logits,
            detach=self.detach_ambiguity,
            temperature=self.ambiguity_temperature,
        )
        return logits, ambiguity

    def fuse(
        self,
        stage: int,
        decoder: torch.Tensor,
        local: torch.Tensor,
        global_feature: torch.Tensor,
        ambiguity: torch.Tensor,
        add_local_skip: bool,
    ) -> torch.Tensor:
        key = str(int(stage))
        if key not in self.fusions:
            return decoder + local if add_local_skip else decoder
        return self.fusions[key](
            decoder,
            local,
            global_feature,
            ambiguity,
            add_local_skip=add_local_skip,
        )

    def refine_stage(
        self,
        stage: int,
        feature: torch.Tensor,
        previous_logits: torch.Tensor,
    ):
        if self.final_refiner is not None:
            return self.stage_refiners[str(int(stage))](feature, previous_logits)

        # With logit refinement off, propagate the coarse prediction so it can
        # still supply ambiguity for the next stage and receive auxiliary loss.
        logits = F.interpolate(
            previous_logits,
            size=feature.shape[1:3],
            mode="bilinear",
            align_corners=False,
        )
        ambiguity = ambiguity_from_logits(
            logits,
            detach=self.detach_ambiguity,
            temperature=self.ambiguity_temperature,
        )
        return logits, ambiguity

    def refine_final(
        self,
        feature: torch.Tensor,
        previous_logits: torch.Tensor,
        base_logits: torch.Tensor,
    ):
        if self.final_refiner is None:
            ambiguity = ambiguity_from_logits(
                base_logits,
                detach=self.detach_ambiguity,
                temperature=self.ambiguity_temperature,
            )
            return base_logits, ambiguity

        # Anchor the last correction to VM-UNet's original segmentation head.
        # With zero correction heads and zero feature scales this is exactly the
        # existing MDK-VMUNet output, which makes ablations well controlled.
        return self.final_refiner(feature, previous_logits, base_logits=base_logits)
