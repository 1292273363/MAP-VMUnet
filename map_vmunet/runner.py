# SPDX-License-Identifier: Apache-2.0
"""Shared training and evaluation runner for the paper configurations."""

import argparse
import csv
import json
from pathlib import Path

from .data import ISICDataset, check_files, read_splits


ROOT = Path(__file__).resolve().parents[1]


def load_config(filename):
    return json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))


def save_test_results(output, scores, rows, test_entries):
    (output / "test_summary.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    with (output / "test_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *rows[0].keys()])
        writer.writeheader()
        for (sample_id, _, _), metrics in zip(test_entries, rows):
            writer.writerow({"sample_id": sample_id, **metrics})
    print(json.dumps(scores, indent=2))


def smoke(model, settings, device, torch):
    from map_vmunet.training import optimizer_for, segmentation, training_loss

    size = settings["input_size"]
    model.train()
    optimizer = optimizer_for(model, settings)
    images = torch.rand(1, 3, *size, device=device) * 255
    masks = torch.randint(0, 2, (1, 1, *size), device=device).float()
    result = model(images)
    if segmentation(result).shape != masks.shape:
        raise RuntimeError("Smoke test prediction and target shapes differ.")
    loss = training_loss(result, masks, settings)
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke test returned non-finite loss.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True
    )
    optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = segmentation(model(images))
    if prediction.shape != masks.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Smoke test returned an invalid evaluation prediction.")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Smoke test passed. Shape {tuple(prediction.shape)}, parameters {parameters:,}, loss {loss.item():.6f}.")


def run_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=("isic2017", "isic2018"), default="isic2018")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pretrained", type=Path, help="VMamba initialization checkpoint")
    parser.add_argument("--output", type=Path, help="New directory for run outputs")
    parser.add_argument("--gpu", type=int, default=0)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-data", action="store_true", help="Check the split lists and all input paths")
    actions.add_argument("--smoke", action="store_true", help="Run one synthetic CUDA optimizer step")
    actions.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Evaluate a trained checkpoint on the test split")
    return parser


def run(parser, args, config):
    settings = config["training"]

    if not (args.smoke or args.check_data or args.evaluate) and args.pretrained is None:
        parser.error("training requires --pretrained with a VMamba checkpoint")
    if not (args.smoke or args.check_data) and args.output is None:
        parser.error("training and evaluation require --output with a new run directory")
    if args.evaluate and args.pretrained:
        parser.error("--evaluate uses the trained checkpoint without --pretrained")
    for path in (args.pretrained, args.evaluate):
        if path is not None and not path.is_file():
            parser.error(f"Checkpoint does not exist: {path}")
    if not args.smoke:
        split_dir = ROOT / "splits" / f"{args.dataset}_70_10_20_seed{settings['split_seed']}"
        splits = read_splits(split_dir, args.data_root, args.dataset)
        check_files({"test": splits["test"]} if args.evaluate else splits)
        if args.check_data:
            print(json.dumps({name: len(entries) for name, entries in splits.items()}))
            return

    import torch
    from torch.utils.data import DataLoader
    from .model import MAPVMUNet
    from .training import evaluate, optimizer_for, set_seed, train_epoch

    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("A CUDA-capable GPU is required. Check --gpu and the installation instructions.")
    device = torch.device("cuda", args.gpu)
    with torch.cuda.device(device):
        set_seed(settings["seed"])
        if not args.smoke:
            test_dataset = ISICDataset(splits["test"], args.dataset, settings["input_size"])
            if not args.evaluate:
                train_dataset = ISICDataset(
                    splits["train"], args.dataset, settings["input_size"], training=True
                )
                val_dataset = ISICDataset(splits["val"], args.dataset, settings["input_size"])

        model = MAPVMUNet(**config["model"])
        if args.pretrained:
            model.load_from(str(args.pretrained))
        model.to(device)
        if args.smoke:
            smoke(model, settings, device, torch)
            return

        output = args.output
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_config.json").write_text(
            json.dumps({
                **config,
                "dataset": args.dataset,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "training_rotation_angle": None if args.evaluate else train_dataset.rotation_angle,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        workers = settings["num_workers"]
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=workers, pin_memory=True)
        checkpoint_path = args.evaluate
        if checkpoint_path is None:
            train_loader = DataLoader(
                train_dataset, batch_size=settings["batch_size"], shuffle=True,
                num_workers=workers, pin_memory=True,
            )
            val_loader = DataLoader(val_dataset, batch_size=1, num_workers=workers, pin_memory=True)
            optimizer = optimizer_for(model, settings)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=settings["t_max"], eta_min=settings["eta_min"]
            )
            checkpoint_path = output / "best_dice.pth"
            best_dice = -1.0
            with (output / "history.jsonl").open("w", encoding="utf-8") as history:
                for epoch in range(1, settings["epochs"] + 1):
                    learning_rates = [group["lr"] for group in optimizer.param_groups]
                    loss = train_epoch(model, train_loader, optimizer, settings, device)
                    scheduler.step()
                    scores, _ = evaluate(model, val_loader, settings["threshold"], device)
                    record = json.dumps({
                        "epoch": epoch, "train_loss": loss, "learning_rates": learning_rates,
                        "validation": scores,
                    })
                    print(record, flush=True)
                    history.write(record + "\n")
                    history.flush()
                    if scores["dice"] > best_dice:
                        best_dice = scores["dice"]
                        pending = checkpoint_path.with_suffix(".tmp")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "config": config["name"],
                            "dataset": args.dataset,
                            "validation": scores,
                        }, pending)
                        pending.replace(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        scores, rows = evaluate(model, test_loader, settings["threshold"], device)
        save_test_results(output, scores, rows, splits["test"])
