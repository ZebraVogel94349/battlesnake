#!/usr/bin/env python3
"""Train a frozen Snake-25 behavioral clone for the V21 opponent league."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from snake25_bc import build_dataset, load_dataset, save_dataset
from train_cuda import (
    ActorCritic,
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_model_state,
    load_model_state,
)


ROTATED_ACTIONS = torch.tensor(
    [
        [0, 1, 2, 3],
        [2, 3, 1, 0],
        [1, 0, 3, 2],
        [3, 2, 0, 1],
    ],
    dtype=torch.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("../game_logs"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path("models/snake25_bc_dataset_v1.pt"),
    )
    parser.add_argument("--snake-name", default="25")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-sequences", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--duel-weight", type=float, default=2.5)
    parser.add_argument("--distill-coef", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=252525)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--rotation-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inputs without building a dataset or training",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {args.source}")
    if not args.logs.is_dir():
        raise FileNotFoundError(f"game log directory does not exist: {args.logs}")
    if args.sequence_length <= 0 or args.batch_sequences <= 0 or args.epochs <= 0:
        raise ValueError("sequence length, batch size, and epochs must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.duel_weight < 1.0:
        raise ValueError("duel weight must be at least one")
    if args.distill_coef < 0.0 or args.max_grad_norm <= 0.0:
        raise ValueError("distill coefficient and grad norm are invalid")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max files must be positive")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_or_build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_cache.is_file() and not args.rebuild_dataset:
        dataset = load_dataset(args.dataset_cache)
        compatible = (
            dataset.get("snake_name") == args.snake_name
            and dataset.get("sequence_length") == args.sequence_length
            and dataset.get("validation_fraction") == args.validation_fraction
            and dataset.get("max_files") == args.max_files
        )
        if compatible:
            print(
                f"[bc:data] loaded {args.dataset_cache}: "
                f"games={dataset['game_count']} "
                f"transitions={dataset['transition_count']} "
                f"duel={dataset['duel_transition_count']}",
                flush=True,
            )
            return dataset
        print(
            f"[bc:data] rebuilding incompatible cache {args.dataset_cache}",
            flush=True,
        )
    dataset = build_dataset(
        args.logs,
        snake_name=args.snake_name,
        sequence_length=args.sequence_length,
        validation_fraction=args.validation_fraction,
        max_files=args.max_files,
    )
    save_dataset(dataset, args.dataset_cache)
    print(
        f"[bc:data] built {args.dataset_cache}: games={dataset['game_count']} "
        f"transitions={dataset['transition_count']} "
        f"duel={dataset['duel_transition_count']} "
        f"train_sequences={len(dataset['train'])} "
        f"validation_sequences={len(dataset['validation'])}",
        flush=True,
    )
    return dataset


def collate(
    sequences: list[dict[str, Any]],
    device: torch.device,
    *,
    augment: bool,
    rng: random.Random,
) -> tuple[torch.Tensor, ...]:
    max_length = max(int(sequence["actions"].shape[0]) for sequence in sequences)
    batch = len(sequences)
    obs = torch.zeros((max_length, batch, 9, 29, 29), dtype=torch.float32)
    actions = torch.zeros((max_length, batch), dtype=torch.int64)
    valid = torch.zeros((max_length, batch), dtype=torch.bool)
    duel = torch.zeros((max_length, batch), dtype=torch.bool)
    coverage = torch.zeros((max_length, batch), dtype=torch.float32)
    for column, sequence in enumerate(sequences):
        length = int(sequence["actions"].shape[0])
        sequence_obs = sequence["obs"].float()
        sequence_actions = sequence["actions"].long()
        if augment:
            rotations = rng.randrange(4)
            if rotations:
                sequence_obs = torch.rot90(sequence_obs, rotations, dims=(-2, -1))
                sequence_actions = ROTATED_ACTIONS[rotations, sequence_actions]
        obs[:length, column] = sequence_obs
        actions[:length, column] = sequence_actions
        valid[:length, column] = True
        duel[:length, column] = sequence["duel"]
        coverage[:length, column] = sequence["coverage"]
    return tuple(
        tensor.to(device, non_blocking=device.type == "cuda")
        for tensor in (obs, actions, valid, duel, coverage)
    )


def actor_logits(model: ActorCritic, obs: torch.Tensor) -> torch.Tensor:
    time_steps, batch = obs.shape[:2]
    h, c = model.initial_actor_state(batch, obs.device)
    outputs = []
    for step in range(time_steps):
        logits, h, c = model.actor_forward(obs[step], h, c)
        outputs.append(logits.float())
    return torch.stack(outputs)


@torch.inference_mode()
def evaluate(
    model: ActorCritic,
    sequences: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "correct": 0.0,
        "count": 0.0,
        "duel_correct": 0.0,
        "duel_count": 0.0,
    }
    rng = random.Random(0)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        obs, actions, valid, duel, _ = collate(
            batch, device, augment=False, rng=rng
        )
        logits = actor_logits(model, obs)
        point_loss = F.cross_entropy(
            logits.reshape(-1, 4), actions.reshape(-1), reduction="none"
        ).reshape_as(actions)
        predictions = logits.argmax(dim=-1)
        totals["loss"] += float(point_loss[valid].sum())
        totals["correct"] += float((predictions[valid] == actions[valid]).sum())
        totals["count"] += int(valid.sum())
        duel_valid = valid & duel
        totals["duel_correct"] += float(
            (predictions[duel_valid] == actions[duel_valid]).sum()
        )
        totals["duel_count"] += int(duel_valid.sum())
    return {
        "nll": totals["loss"] / max(1.0, totals["count"]),
        "accuracy": totals["correct"] / max(1.0, totals["count"]),
        "duel_accuracy": totals["duel_correct"] / max(1.0, totals["duel_count"]),
        "count": totals["count"],
        "duel_count": totals["duel_count"],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.check:
        source_payload = torch.load(args.source, map_location="cpu", weights_only=False)
        source_state = checkpoint_model_state(source_payload)
        if not source_state or not isinstance(source_payload.get("optimizer_state_dict"), dict):
            raise ValueError("source is not a resumable training checkpoint")
        source_update = int(source_payload.get("update", -1))
        if args.output.is_file():
            clone_payload = torch.load(
                args.output, map_location="cpu", weights_only=False
            )
            if not clone_payload.get("behavioral_clone"):
                raise ValueError("existing BC output is not a behavioral clone")
            if int(clone_payload.get("update", -2)) != source_update:
                raise ValueError("existing BC output was trained from another update")
            checkpoint_model_state(clone_payload)
        print(f"[bc:check] source={args.source.resolve()}")
        print(f"[bc:check] source_update={source_update}")
        print(f"[bc:check] logs={args.logs.resolve()}")
        print(
            f"[bc:check] output={args.output.resolve()} "
            f"status={'ready' if args.output.is_file() else 'will-be-created'}"
        )
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dataset = load_or_build_dataset(args)
    source_payload = torch.load(args.source, map_location="cpu", weights_only=False)
    model = ActorCritic()
    load_model_state(model, checkpoint_model_state(source_payload), allow_legacy=False)
    source_update = int(source_payload.get("update", 0))
    source_steps = int(source_payload.get("total_steps", 0))
    del source_payload
    model.to(device)
    base_model = copy.deepcopy(model).to(device).eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    for module in (model.critic_cnn, model.critic_lstm, model.critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    actor_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    baseline = evaluate(
        model, dataset["validation"], args.batch_sequences, device
    )
    print(
        f"[bc] device={device} source_update={source_update} "
        f"baseline_nll={baseline['nll']:.4f} "
        f"baseline_acc={baseline['accuracy']:.3f} "
        f"baseline_duel_acc={baseline['duel_accuracy']:.3f}",
        flush=True,
    )
    rng = random.Random(args.seed)
    best_nll = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(dataset["train"])))
        rng.shuffle(order)
        loss_sum = example_sum = 0.0
        for start in range(0, len(order), args.batch_sequences):
            batch = [
                dataset["train"][index]
                for index in order[start : start + args.batch_sequences]
            ]
            obs, actions, valid, duel, coverage = collate(
                batch,
                device,
                augment=args.rotation_augmentation,
                rng=rng,
            )
            logits = actor_logits(model, obs)
            with torch.no_grad():
                base_logits = actor_logits(base_model, obs)
            ce = F.cross_entropy(
                logits.reshape(-1, 4), actions.reshape(-1), reduction="none"
            ).reshape_as(actions)
            confidence = 0.5 + 0.5 * coverage.clamp(0.0, 1.0)
            weights = confidence * torch.where(
                duel,
                torch.full_like(confidence, args.duel_weight),
                torch.ones_like(confidence),
            )
            weights = weights * valid
            imitation_loss = (ce * weights).sum() / weights.sum().clamp_min(1.0)
            base_probs = F.softmax(base_logits, dim=-1)
            distillation = (
                base_probs
                * (F.log_softmax(base_logits, dim=-1) - F.log_softmax(logits, dim=-1))
            ).sum(dim=-1)
            distillation_loss = distillation[valid].mean()
            loss = imitation_loss + args.distill_coef * distillation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm)
            optimizer.step()
            valid_count = int(valid.sum())
            loss_sum += float(imitation_loss.detach()) * valid_count
            example_sum += valid_count

        metrics = evaluate(
            model, dataset["validation"], args.batch_sequences, device
        )
        print(
            f"[bc] epoch={epoch:02d} train_nll={loss_sum / max(1.0, example_sum):.4f} "
            f"val_nll={metrics['nll']:.4f} val_acc={metrics['accuracy']:.3f} "
            f"val_duel_acc={metrics['duel_accuracy']:.3f}",
            flush=True,
        )
        if metrics["nll"] < best_nll - 1e-5:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            print(f"[bc] early stop after epoch {epoch}", flush=True)
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("behavioral cloning produced no checkpoint")
    target = args.output.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    metadata = {
        "kind": "snake25_behavioral_clone",
        "snake_name": args.snake_name,
        "source": str(args.source.resolve()),
        "source_update": source_update,
        "source_total_steps": source_steps,
        "dataset_cache": str(args.dataset_cache.resolve()),
        "dataset_games": dataset["game_count"],
        "dataset_transitions": dataset["transition_count"],
        "dataset_duel_transitions": dataset["duel_transition_count"],
        "best_epoch": best_epoch,
        "baseline": baseline,
        "validation": best_metrics,
        "proxy_observation_version": dataset["proxy_observation_version"],
    }
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "activation": "tanh",
            "separate_critic_lstm": True,
            "behavioral_clone": True,
            "update": source_update,
            "total_steps": source_steps,
            "model_state_dict": best_state,
            "bc_metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, target)
    metrics_path = target.with_suffix(".json")
    metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    metrics_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(metrics_tmp, metrics_path)
    print(
        f"[bc] saved {target}: epoch={best_epoch} "
        f"accuracy={best_metrics['accuracy']:.3f} "
        f"duel_accuracy={best_metrics['duel_accuracy']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
