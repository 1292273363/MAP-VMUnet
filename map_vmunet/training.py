# SPDX-License-Identifier: Apache-2.0
# Adapted from VM-UNet's BCE/Dice training and the MAP-VMUnet fixed-split protocol.
"""Shared training and evaluation for the three released configurations."""

import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def optimizer_for(model, settings):
    backbone_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = new_parameters if name.startswith(("mdk_blocks.", "pard.")) else backbone_parameters
            group.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": settings["backbone_lr"]},
            {"params": new_parameters, "lr": settings["new_module_lr"]},
        ],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
        amsgrad=settings["amsgrad"],
    )


def segmentation(output):
    return output["seg"] if isinstance(output, dict) else output


def bce_dice(prediction, target):
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    intersection = (prediction * target).sum(dim=1)
    dice = (2 * intersection + 1) / (prediction.sum(dim=1) + target.sum(dim=1) + 1)
    return F.binary_cross_entropy(prediction, target) + (1 - dice.sum() / prediction.shape[0])


def training_loss(output, target, settings):
    loss = bce_dice(segmentation(output), target)
    if settings["lambda_aux"] == 0:
        return loss

    auxiliary = output["aux_seg"]
    weights = settings["aux_weights"]
    if len(weights) != len(auxiliary):
        raise ValueError("The auxiliary weights must match the number of PARD predictions.")
    auxiliary_loss = target.new_zeros(())
    for weight, prediction in zip(weights, auxiliary):
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        auxiliary_loss = auxiliary_loss + weight * bce_dice(prediction, target)
    return loss + settings["lambda_aux"] * (auxiliary_loss / sum(weights))


def train_epoch(model, loader, optimizer, settings, device):
    model.train()
    losses = []
    for images, masks in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad()
        loss = training_loss(model(images), masks, settings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=settings["grad_clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def binary_metrics(tp, tn, fp, fn):
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, threshold, device):
    model.eval()
    rows = []
    counts = []
    for images, masks in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        prediction = segmentation(model(images))
        if prediction.shape != masks.shape:
            raise ValueError(f"Prediction and mask shapes differ: {prediction.shape} versus {masks.shape}")
        prediction = prediction.flatten(1) >= threshold
        target = masks.flatten(1) >= 0.5
        batch_counts = torch.stack(
            (
                (prediction & target).sum(1),
                (~prediction & ~target).sum(1),
                (prediction & ~target).sum(1),
                (~prediction & target).sum(1),
            ),
            dim=1,
        ).tolist()
        counts.extend(batch_counts)
        for tp, tn, fp, fn in batch_counts:
            rows.append({**binary_metrics(tp, tn, fp, fn), "tp": tp, "tn": tn, "fp": fp, "fn": fn})

    global_scores = binary_metrics(*(sum(values) for values in zip(*counts)))
    scores = dict(global_scores)
    scores.update(
        global_iou=global_scores["iou"],
        global_dice=global_scores["dice"],
        iou=sum(row["iou"] for row in rows) / len(rows),
        dice=sum(row["dice"] for row in rows) / len(rows),
    )
    return scores, rows
