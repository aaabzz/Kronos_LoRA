"""Core teacher/predicted/mixed conditioning training for Kronos QV LoRA."""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import random
import re
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lora import (
    build_full_finetune_model,
    build_qv_lora_model,
    load_frozen_tokenizer,
    load_full_finetune_model,
)
from lora.config import LORA_TARGET_PROFILES, PROJECT_ROOT, apply_target_profile
from lora.model_factory import initialize_qv_lora_model, load_qv_lora_adapter

from .config import ExperimentConfig
from .dataset import build_datasets
from .sharding import (
    DistributedSequentialSampler,
    DistributedShardSampler,
    ShardShuffleSampler,
    TokenShardDataset,
)


@dataclass
class TokenPredictionMetrics:
    top1_correct: int
    top5_correct: int
    entropy_sum: float


@dataclass
class LossMetrics:
    loss: torch.Tensor
    s1_loss: torch.Tensor
    s2_loss: torch.Tensor
    target_count: int
    s1_prediction: TokenPredictionMetrics | None = None
    s2_prediction: TokenPredictionMetrics | None = None
    teacher_s2_loss: torch.Tensor | None = None
    predicted_s2_loss: torch.Tensor | None = None
    teacher_s2_prediction: TokenPredictionMetrics | None = None


PREDICTION_METRIC_KEYS = (
    "s1_top1_accuracy",
    "s1_top5_accuracy",
    "s2_top1_accuracy",
    "s2_top5_accuracy",
    "s1_mean_entropy",
    "s2_mean_entropy",
)

MIXED_CONDITIONING_METRIC_KEYS = (
    "mixed_loss",
    "predicted_s2_loss",
    "mixed_s2_loss",
    "teacher_s2_loss",
    "teacher_s2_top1_accuracy",
    "teacher_s2_top5_accuracy",
    "teacher_s2_mean_entropy",
    "predicted_minus_teacher_s2_loss_delta",
    "predicted_minus_teacher_s2_top1_accuracy_delta",
)


def lora_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return pre-clipping L2 gradient norms for diagnostic module groups."""
    squared = {
        "grad_norm_total": None,
        "grad_norm_proj_s1": None,
        "grad_norm_transformer": None,
        "grad_norm_dependency": None,
        "grad_norm_other": None,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        group = "grad_norm_other"
        if "head.proj_s1" in name:
            group = "grad_norm_proj_s1"
        elif "dep_layer" in name:
            group = "grad_norm_dependency"
        elif "transformer" in name:
            group = "grad_norm_transformer"
        squared["grad_norm_total"] = (
            contribution
            if squared["grad_norm_total"] is None
            else squared["grad_norm_total"] + contribution
        )
        squared[group] = (
            contribution
            if squared[group] is None
            else squared[group] + contribution
        )
    return {
        key: 0.0 if value is None else math.sqrt(value.item())
        for key, value in squared.items()
    }


def shared_transformer_parameters(
    model: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """Return trainable parameters in the shared Transformer only."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "transformer" in name
        and "dep_layer" not in name
        and "head.proj_s1" not in name
    ]


def flatten_parameter_gradients(
    parameters: list[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    """Flatten parameter gradients into one FP32 vector without changing them."""
    if not parameters:
        raise RuntimeError("No trainable shared Transformer parameters were found.")

    gradients = []
    for _, parameter in parameters:
        if parameter.grad is None:
            gradients.append(
                torch.zeros(
                    parameter.numel(),
                    device=parameter.device,
                    dtype=torch.float32,
                )
            )
        else:
            gradients.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(gradients)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_distributed(requested_device: str) -> DistributedContext:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(False, 0, 0, 1, select_device(requested_device))
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-process DDP training requires CUDA/NCCL.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        True,
        rank,
        local_rank,
        world_size,
        torch.device("cuda", local_rank),
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve auto precision and reject unsupported explicit BF16 requests."""
    if requested == "fp32":
        return "fp32"

    bf16_supported = False
    if device.type == "cuda" and torch.cuda.is_available():
        # is_bf16_supported() checks the current CUDA device, so temporarily
        # select the explicitly requested GPU (for example cuda:1).
        with torch.cuda.device(device):
            bf16_supported = torch.cuda.is_bf16_supported()
    if requested == "bf16" and not bf16_supported:
        raise RuntimeError(
            "BF16 training requires a CUDA device with BF16 support. "
            "Use --precision fp32 on this machine, or --precision auto to fall back."
        )
    return "bf16" if bf16_supported else "fp32"


def build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
):
    """Build optional linear warmup followed by optimizer-step cosine decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be smaller than the total optimizer steps "
            f"({total_steps})."
        )
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            progress = step / warmup_steps
            return 1e-3 + (1.0 - 1e-3) * progress
        decay_steps = total_steps - warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def autocast_context(device: torch.device, precision: str):
    """Create the model-forward autocast context without affecting tokenization."""
    if precision == "bf16":
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        )
    return nullcontext()


def hierarchical_token_loss(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_targets: torch.Tensor,
    s2_targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    s1_weight: float = 0.5,
    s2_weight: float = 0.5,
    compute_prediction_metrics: bool = False,
) -> LossMetrics:
    """Compute s1/s2 cross-entropy only at chronological split targets."""
    mask = loss_mask.bool()
    target_count = int(mask.sum().item())
    if target_count == 0:
        raise ValueError("A batch must contain at least one unmasked target.")

    # Cross entropy is accumulated in FP32 even when the model forward uses BF16.
    s1_loss = F.cross_entropy(s1_logits[mask].float(), s1_targets[mask])
    s2_loss = F.cross_entropy(s2_logits[mask].float(), s2_targets[mask])
    weight_sum = s1_weight + s2_weight
    if s1_weight < 0 or s2_weight < 0 or weight_sum <= 0:
        raise ValueError("Loss weights must be non-negative with a positive sum.")
    loss = (s1_weight * s1_loss + s2_weight * s2_loss) / weight_sum
    s1_prediction = None
    s2_prediction = None
    if compute_prediction_metrics:
        s1_prediction = token_prediction_metrics(s1_logits, s1_targets, mask)
        s2_prediction = token_prediction_metrics(s2_logits, s2_targets, mask)
    return LossMetrics(
        loss,
        s1_loss,
        s2_loss,
        target_count,
        s1_prediction,
        s2_prediction,
    )


def token_prediction_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> TokenPredictionMetrics:
    """Return masked top-k correct counts and entropy sum for one token level."""
    with torch.no_grad():
        selected_logits = logits[mask].float()
        selected_targets = targets[mask]
        top_k = min(5, selected_logits.shape[-1])
        predictions = selected_logits.topk(top_k, dim=-1).indices
        top1_correct = int(
            predictions[:, 0].eq(selected_targets).sum().item()
        )
        top5_correct = int(
            predictions.eq(selected_targets.unsqueeze(-1)).any(dim=-1).sum().item()
        )
        log_probabilities = F.log_softmax(selected_logits, dim=-1)
        entropy_sum = float(
            -(log_probabilities.exp() * log_probabilities).sum().item()
        )
    return TokenPredictionMetrics(top1_correct, top5_correct, entropy_sum)


def prepare_batch(
    tokenizer,
    batch,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Move/tokenize one batch and return aligned model inputs and targets."""
    if len(batch) == 3:
        features, stamps, loss_mask = batch
        features = features.to(device, non_blocking=True)
        stamps = stamps.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        if tokenizer is None:
            raise RuntimeError("Raw Parquet batches require a tokenizer.")
        # The frozen tokenizer produces discrete supervision labels, so keep
        # its computation in FP32 and outside the model autocast region.
        with torch.no_grad():
            s1_tokens, s2_tokens = tokenizer.encode(features, half=True)

        s1_input, s1_targets = s1_tokens[:, :-1], s1_tokens[:, 1:]
        s2_input, s2_targets = s2_tokens[:, :-1], s2_tokens[:, 1:]
        stamps = stamps[:, :-1, :]
    elif len(batch) == 6:
        (
            s1_input,
            s2_input,
            s1_targets,
            s2_targets,
            stamps,
            loss_mask,
        ) = (
            value.to(device, non_blocking=True) for value in batch
        )
    else:
        raise ValueError(f"Expected a raw or tokenized batch, got {len(batch)} items.")

    return s1_input, s2_input, s1_targets, s2_targets, stamps, loss_mask


def forward_batch(
    model,
    tokenizer,
    batch,
    device: torch.device,
    *,
    precision: str = "fp32",
    s1_loss_weight: float = 0.5,
    s2_loss_weight: float = 0.5,
    s2_conditioning: str = "teacher",
    s2_teacher_loss_weight: float = 0.1,
    s2_predicted_loss_weight: float = 0.9,
    s2_top_k: int = 5,
    s2_temperature: float = 1.0,
    compute_prediction_metrics: bool = False,
) -> LossMetrics:
    (
        s1_input,
        s2_input,
        s1_targets,
        s2_targets,
        stamps,
        loss_mask,
    ) = prepare_batch(tokenizer, batch, device)

    with autocast_context(device, precision):
        model_kwargs = {
            "use_teacher_forcing": s2_conditioning == "teacher",
            "s1_targets": s1_targets,
        }
        # Preserve the exact legacy call signature for the default teacher
        # path. Custom wrappers written before predicted-s1 training therefore
        # remain compatible.
        if s2_conditioning != "teacher":
            model_kwargs["s2_conditioning"] = s2_conditioning
        if "topk" in s2_conditioning:
            model_kwargs["s2_top_k"] = s2_top_k
            model_kwargs["s2_temperature"] = s2_temperature
        outputs = model(
            s1_input,
            s2_input,
            stamps,
            **model_kwargs,
        )
        if not s2_conditioning.startswith("mixed-"):
            s1_logits, s2_logits = outputs
            return hierarchical_token_loss(
                s1_logits,
                s2_logits,
                s1_targets,
                s2_targets,
                loss_mask,
                s1_weight=s1_loss_weight,
                s2_weight=s2_loss_weight,
                compute_prediction_metrics=compute_prediction_metrics,
            )

        s1_logits, predicted_s2_logits, teacher_s2_logits = outputs
        predicted_metrics = hierarchical_token_loss(
            s1_logits,
            predicted_s2_logits,
            s1_targets,
            s2_targets,
            loss_mask,
            s1_weight=s1_loss_weight,
            s2_weight=s2_loss_weight,
            compute_prediction_metrics=compute_prediction_metrics,
        )
        mask = loss_mask.bool()
        teacher_s2_loss = F.cross_entropy(
            teacher_s2_logits[mask].float(),
            s2_targets[mask],
        )
        branch_weight_sum = (
            s2_teacher_loss_weight + s2_predicted_loss_weight
        )
        if (
            not math.isfinite(s2_teacher_loss_weight)
            or not math.isfinite(s2_predicted_loss_weight)
            or s2_teacher_loss_weight <= 0
            or s2_predicted_loss_weight <= 0
        ):
            raise ValueError(
                "Mixed s2 loss weights must both be finite and positive."
            )
        mixed_s2_loss = (
            s2_teacher_loss_weight * teacher_s2_loss
            + s2_predicted_loss_weight * predicted_metrics.s2_loss
        ) / branch_weight_sum
        overall_weight_sum = s1_loss_weight + s2_loss_weight
        mixed_loss = (
            s1_loss_weight * predicted_metrics.s1_loss
            + s2_loss_weight * mixed_s2_loss
        ) / overall_weight_sum
        teacher_prediction = (
            token_prediction_metrics(teacher_s2_logits, s2_targets, mask)
            if compute_prediction_metrics
            else None
        )
        return LossMetrics(
            loss=mixed_loss,
            s1_loss=predicted_metrics.s1_loss,
            s2_loss=mixed_s2_loss,
            target_count=predicted_metrics.target_count,
            s1_prediction=predicted_metrics.s1_prediction,
            s2_prediction=predicted_metrics.s2_prediction,
            teacher_s2_loss=teacher_s2_loss,
            predicted_s2_loss=predicted_metrics.s2_loss,
            teacher_s2_prediction=teacher_prediction,
        )


def _probe_task_gradient(
    model: nn.Module,
    tokenizer,
    batch,
    device: torch.device,
    *,
    precision: str,
    task: str,
    s2_conditioning: str,
    s2_teacher_loss_weight: float,
    s2_predicted_loss_weight: float,
    s2_top_k: int,
    s2_temperature: float,
    shared_parameters: list[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    """Replay one batch and return one task's local shared gradient vector."""
    model.zero_grad(set_to_none=True)
    sync_context = (
        model.no_sync()
        if isinstance(model, DistributedDataParallel)
        else nullcontext()
    )
    with sync_context:
        metrics = forward_batch(
            model,
            tokenizer,
            batch,
            device,
            precision=precision,
            s2_conditioning=s2_conditioning,
            s2_teacher_loss_weight=s2_teacher_loss_weight,
            s2_predicted_loss_weight=s2_predicted_loss_weight,
            s2_top_k=s2_top_k,
            s2_temperature=s2_temperature,
        )
        if task == "s1":
            metrics.s1_loss.backward()
        elif task == "s2":
            metrics.s2_loss.backward()
        else:
            raise ValueError(f"Unknown gradient-probe task: {task!r}.")
    return flatten_parameter_gradients(shared_parameters)


def task_gradient_conflict_metrics(
    model: nn.Module,
    tokenizer,
    batches: list[Any],
    device: torch.device,
    *,
    precision: str,
    distributed: DistributedContext,
    s1_weight: float,
    s2_weight: float,
    s2_conditioning: str = "teacher",
    s2_teacher_loss_weight: float = 0.1,
    s2_predicted_loss_weight: float = 0.9,
    s2_top_k: int = 5,
    s2_temperature: float = 1.0,
) -> dict[str, float]:
    """Compare accumulated s1/s2 gradients on shared Transformer weights.

    The training accumulation group is replayed after its optimizer update. Each
    task gradient is averaged over local microbatches and then over DDP ranks.
    The isolated replays never write an optimizer step.
    """
    if not batches:
        raise ValueError("Gradient conflict diagnostics require at least one batch.")
    shared_parameters = shared_transformer_parameters(model)
    s1_gradient: torch.Tensor | None = None
    s2_gradient: torch.Tensor | None = None
    batch_scale = 1.0 / len(batches)

    cuda_devices = (
        [device.index]
        if device.type == "cuda" and device.index is not None
        else []
    )
    # The outer fork prevents diagnostics from advancing the training RNG. For
    # each pair, restoring the starting state gives s1 and s2 identical dropout
    # masks; the next batch still receives a different mask.
    with torch.random.fork_rng(devices=cuda_devices):
        for batch in batches:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None
            )
            current_s1 = _probe_task_gradient(
                model,
                tokenizer,
                batch,
                device,
                precision=precision,
                task="s1",
                s2_conditioning=s2_conditioning,
                s2_teacher_loss_weight=s2_teacher_loss_weight,
                s2_predicted_loss_weight=s2_predicted_loss_weight,
                s2_top_k=s2_top_k,
                s2_temperature=s2_temperature,
                shared_parameters=shared_parameters,
            )
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
            current_s2 = _probe_task_gradient(
                model,
                tokenizer,
                batch,
                device,
                precision=precision,
                task="s2",
                s2_conditioning=s2_conditioning,
                s2_teacher_loss_weight=s2_teacher_loss_weight,
                s2_predicted_loss_weight=s2_predicted_loss_weight,
                s2_top_k=s2_top_k,
                s2_temperature=s2_temperature,
                shared_parameters=shared_parameters,
            )
            if s1_gradient is None:
                s1_gradient = current_s1.mul(batch_scale)
                s2_gradient = current_s2.mul(batch_scale)
            else:
                s1_gradient.add_(current_s1, alpha=batch_scale)
                s2_gradient.add_(current_s2, alpha=batch_scale)

    model.zero_grad(set_to_none=True)
    assert s1_gradient is not None and s2_gradient is not None
    if distributed.enabled:
        dist.all_reduce(s1_gradient, op=dist.ReduceOp.SUM)
        dist.all_reduce(s2_gradient, op=dist.ReduceOp.SUM)
        s1_gradient.div_(distributed.world_size)
        s2_gradient.div_(distributed.world_size)

    s1_norm = torch.linalg.vector_norm(s1_gradient)
    s2_norm = torch.linalg.vector_norm(s2_gradient)
    norm_product = s1_norm * s2_norm
    cosine = (
        torch.dot(s1_gradient, s2_gradient) / norm_product
        if norm_product.item() > 0.0
        else torch.zeros((), device=device)
    )
    weight_sum = s1_weight + s2_weight
    effective_s1_norm = s1_norm * (s1_weight / weight_sum)
    effective_s2_norm = s2_norm * (s2_weight / weight_sum)

    return {
        "transformer_s1_grad_norm": float(s1_norm.item()),
        "transformer_s2_grad_norm": float(s2_norm.item()),
        "transformer_grad_cosine": float(cosine.clamp(-1.0, 1.0).item()),
        "transformer_s2_to_s1_ratio": float(
            s2_norm.item() / (s1_norm.item() + 1e-12)
        ),
        "effective_s2_to_s1_ratio": float(
            effective_s2_norm.item() / (effective_s1_norm.item() + 1e-12)
        ),
    }


def evaluate(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    epoch: int | None = None,
    precision: str = "fp32",
    s1_loss_weight: float = 0.5,
    s2_loss_weight: float = 0.5,
    s2_conditioning: str = "teacher",
    s2_teacher_loss_weight: float = 0.1,
    s2_predicted_loss_weight: float = 0.9,
    s2_top_k: int = 5,
    s2_temperature: float = 1.0,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    """Evaluate one split and return globally target-weighted losses/metrics."""

    model.eval()
    weighted_loss = 0.0
    weighted_s1 = 0.0
    weighted_s2 = 0.0
    weighted_mixed_s2 = 0.0
    weighted_teacher_s2 = 0.0
    s1_top1_correct = 0
    s1_top5_correct = 0
    s1_entropy_sum = 0.0
    s2_top1_correct = 0
    s2_top5_correct = 0
    s2_entropy_sum = 0.0
    teacher_s2_top1_correct = 0
    teacher_s2_top5_correct = 0
    teacher_s2_entropy_sum = 0.0
    total_targets = 0
    is_mixed = s2_conditioning.startswith("mixed-")
    cuda_devices = (
        [device.index]
        if device.type == "cuda" and device.index is not None
        else []
    )
    rng_context = (
        torch.random.fork_rng(devices=cuda_devices)
        if s2_conditioning in {"sample", "mixed-sample"}
        else nullcontext()
    )
    # Sample-conditioned validation must not advance the training RNG; changing
    # validation_interval should never change subsequent optimizer updates.
    with rng_context, torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            metrics = forward_batch(
                model,
                tokenizer,
                batch,
                device,
                precision=precision,
                s1_loss_weight=s1_loss_weight,
                s2_loss_weight=s2_loss_weight,
                s2_conditioning=s2_conditioning,
                s2_teacher_loss_weight=s2_teacher_loss_weight,
                s2_predicted_loss_weight=s2_predicted_loss_weight,
                s2_top_k=s2_top_k,
                s2_temperature=s2_temperature,
                compute_prediction_metrics=True,
            )
            if not torch.isfinite(metrics.loss):
                if distributed is None or not distributed.enabled:
                    epoch_text = f", epoch={epoch}" if epoch is not None else ""
                    raise FloatingPointError(
                        f"Non-finite {split} loss at "
                        f"batch={batch_index}{epoch_text}."
                    )
                weighted_loss = float("nan")
                weighted_s1 = float("nan")
                weighted_s2 = float("nan")
                continue
            count = metrics.target_count
            weighted_s1 += metrics.s1_loss.item() * count
            if is_mixed:
                if (
                    metrics.teacher_s2_loss is None
                    or metrics.predicted_s2_loss is None
                    or metrics.teacher_s2_prediction is None
                ):
                    raise RuntimeError("Mixed conditioning metrics were not computed.")
                # Select checkpoints using the inference-like predicted branch,
                # not the teacher-weighted training objective.
                selection_loss = (
                    s1_loss_weight * metrics.s1_loss
                    + s2_loss_weight * metrics.predicted_s2_loss
                ) / (s1_loss_weight + s2_loss_weight)
                weighted_loss += selection_loss.item() * count
                weighted_s2 += metrics.predicted_s2_loss.item() * count
                weighted_mixed_s2 += metrics.s2_loss.item() * count
                weighted_teacher_s2 += metrics.teacher_s2_loss.item() * count
                teacher_s2_top1_correct += (
                    metrics.teacher_s2_prediction.top1_correct
                )
                teacher_s2_top5_correct += (
                    metrics.teacher_s2_prediction.top5_correct
                )
                teacher_s2_entropy_sum += (
                    metrics.teacher_s2_prediction.entropy_sum
                )
            else:
                weighted_loss += metrics.loss.item() * count
                weighted_s2 += metrics.s2_loss.item() * count
            if metrics.s1_prediction is None or metrics.s2_prediction is None:
                raise RuntimeError("Evaluation prediction metrics were not computed.")
            s1_top1_correct += metrics.s1_prediction.top1_correct
            s1_top5_correct += metrics.s1_prediction.top5_correct
            s1_entropy_sum += metrics.s1_prediction.entropy_sum
            s2_top1_correct += metrics.s2_prediction.top1_correct
            s2_top5_correct += metrics.s2_prediction.top5_correct
            s2_entropy_sum += metrics.s2_prediction.entropy_sum
            total_targets += count
    if distributed is not None and distributed.enabled:
        totals = torch.tensor(
            [
                weighted_loss,
                weighted_s1,
                weighted_s2,
                weighted_mixed_s2,
                weighted_teacher_s2,
                s1_top1_correct,
                s1_top5_correct,
                s1_entropy_sum,
                s2_top1_correct,
                s2_top5_correct,
                s2_entropy_sum,
                teacher_s2_top1_correct,
                teacher_s2_top5_correct,
                teacher_s2_entropy_sum,
                total_targets,
            ],
            dtype=torch.float64,
            device=distributed.device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (
            weighted_loss,
            weighted_s1,
            weighted_s2,
            weighted_mixed_s2,
            weighted_teacher_s2,
            s1_top1_correct,
            s1_top5_correct,
            s1_entropy_sum,
            s2_top1_correct,
            s2_top5_correct,
            s2_entropy_sum,
            teacher_s2_top1_correct,
            teacher_s2_top5_correct,
            teacher_s2_entropy_sum,
            total_targets_float,
        ) = totals.tolist()
        total_targets = int(total_targets_float)
    if not math.isfinite(weighted_loss):
        epoch_text = f", epoch={epoch}" if epoch is not None else ""
        rank_text = (
            f", rank={distributed.rank}"
            if distributed is not None and distributed.enabled
            else ""
        )
        raise FloatingPointError(f"Non-finite {split} loss{epoch_text}{rank_text}.")
    if total_targets == 0:
        raise RuntimeError(f"{split.capitalize()} loader produced no targets.")
    result = {
        "loss": weighted_loss / total_targets,
        "s1_loss": weighted_s1 / total_targets,
        "s2_loss": weighted_s2 / total_targets,
        "s1_top1_accuracy": s1_top1_correct / total_targets,
        "s1_top5_accuracy": s1_top5_correct / total_targets,
        "s2_top1_accuracy": s2_top1_correct / total_targets,
        "s2_top5_accuracy": s2_top5_correct / total_targets,
        "s1_mean_entropy": s1_entropy_sum / total_targets,
        "s2_mean_entropy": s2_entropy_sum / total_targets,
        "targets": float(total_targets),
    }
    if is_mixed:
        result.update(
            {
                "predicted_s2_loss": result["s2_loss"],
                "mixed_s2_loss": weighted_mixed_s2 / total_targets,
                "teacher_s2_loss": weighted_teacher_s2 / total_targets,
                "teacher_s2_top1_accuracy": (
                    teacher_s2_top1_correct / total_targets
                ),
                "teacher_s2_top5_accuracy": (
                    teacher_s2_top5_correct / total_targets
                ),
                "teacher_s2_mean_entropy": (
                    teacher_s2_entropy_sum / total_targets
                ),
                "predicted_minus_teacher_s2_loss_delta": (
                    result["s2_loss"] - weighted_teacher_s2 / total_targets
                ),
                "predicted_minus_teacher_s2_top1_accuracy_delta": (
                    result["s2_top1_accuracy"]
                    - teacher_s2_top1_correct / total_targets
                ),
            }
        )
        result["mixed_loss"] = (
            s1_loss_weight * result["s1_loss"]
            + s2_loss_weight * result["mixed_s2_loss"]
        ) / (s1_loss_weight + s2_loss_weight)
    return result


def should_run_step_validation(
    global_step: int,
    validation_interval: int,
    *,
    is_epoch_end: bool,
) -> bool:
    """Return whether this optimizer boundary needs an extra validation pass."""

    return (
        validation_interval > 0
        and global_step % validation_interval == 0
        and not is_epoch_end
    )


def is_significant_validation_improvement(
    candidate_loss: float,
    reference_loss: float,
    min_delta: float,
) -> bool:
    """Return whether a validation decrease is large enough to reset patience."""
    return (
        candidate_loss < reference_loss
        and reference_loss - candidate_loss >= min_delta
    )


def create_loaders(config: ExperimentConfig, distributed: DistributedContext):
    datasets, boundaries = build_datasets(config.data)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_dataset = datasets["train"]
    if distributed.enabled and isinstance(train_dataset, TokenShardDataset):
        train_sampler = DistributedShardSampler(
            train_dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            seed=config.training.seed,
        )
    elif isinstance(train_dataset, TokenShardDataset):
        train_sampler = ShardShuffleSampler(train_dataset, seed=config.training.seed)
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=config.training.seed,
        )

    validation_sampler = (
        DistributedSequentialSampler(
            datasets["validation"],
            num_replicas=distributed.world_size,
            rank=distributed.rank,
        )
        if distributed.enabled
        else None
    )
    test_sampler = (
        DistributedSequentialSampler(
            datasets["test"],
            num_replicas=distributed.world_size,
            rank=distributed.rank,
        )
        if distributed.enabled
        else None
    )
    loaders = {
        "train": DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                config.training.seed + distributed.rank
            ),
            **common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            sampler=validation_sampler,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                config.training.seed + distributed.rank + 10_000
            ),
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                config.training.seed + distributed.rank + 20_000
            ),
            **common,
        ),
    }
    return loaders, datasets, boundaries


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def deployment_metadata(
    config: ExperimentConfig,
    run_dir: Path,
    data_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Build portable metadata consumed by API adapter discovery."""
    base_name = config.lora.base_model_path.name
    artifact_label = "LoRA" if config.training.finetune_mode == "lora" else "Full"
    display_name = config.training.model_display_name or (
        f"{run_dir.name} {artifact_label} · {base_name}"
    )
    target_profile = next(
        (
            profile
            for profile in LORA_TARGET_PROFILES
            if apply_target_profile(config.lora, profile).target_modules
            == config.lora.target_modules
            and apply_target_profile(config.lora, profile).exclude_modules
            == config.lora.exclude_modules
        ),
        "custom",
    )
    base_config_path = config.lora.base_model_path / "config.json"
    return {
        "format_version": 1,
        "model_key": f"{config.training.finetune_mode}:{run_dir.name}",
        "display_name": display_name,
        "base_model": {
            "key": base_name.removeprefix("Kronos-").lower(),
            "name": base_name,
            "path": _portable_project_path(config.lora.base_model_path),
            "config_sha256": _sha256_file(base_config_path),
        },
        "tokenizer": {
            "name": config.lora.tokenizer_path.name,
            "path": _portable_project_path(config.lora.tokenizer_path),
            "fingerprint": data_fingerprint.get("tokenizer_fingerprint"),
        },
        "adapter": {
            "type": config.training.finetune_mode,
            "target_profile": (
                target_profile if config.training.finetune_mode == "lora" else None
            ),
            "rank": config.lora.rank if config.training.finetune_mode == "lora" else None,
            "alpha": config.lora.alpha if config.training.finetune_mode == "lora" else None,
            "full_finetune_proj_s1": (
                config.lora.full_finetune_proj_s1
                if config.training.finetune_mode == "lora"
                else True
            ),
        },
        "training": {
            "run_name": run_dir.name,
            "target_mode": config.data.target_mode,
            "target_length": config.data.target_length,
        },
    }


def save_deployable_model(
    model,
    directory: Path,
    config: ExperimentConfig,
    run_dir: Path,
    data_fingerprint: dict[str, Any],
) -> None:
    """Save PEFT or full weights together with deployment metadata."""
    model.save_pretrained(directory)
    save_json(
        directory / "deployment.json",
        deployment_metadata(config, run_dir, data_fingerprint),
    )


# Backward-compatible name for callers introduced before full fine-tuning.
save_deployable_adapter = save_deployable_model


def dataset_fingerprint(config: ExperimentConfig) -> dict[str, Any]:
    """Return inexpensive metadata that detects an incompatible data cache."""
    if config.data.token_cache_path is not None:
        manifest_path = config.data.token_cache_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "kind": "token_cache",
            "path": str(config.data.token_cache_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "shard_count": len(manifest.get("shards", [])),
            "counts": manifest.get("counts", {}),
            "tokenizer_fingerprint": manifest.get("tokenizer_fingerprint"),
        }

    from .dataset import discover_parquet_files

    files = discover_parquet_files(config.data.data_path)
    inventory = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "parquet_inventory",
        "path": str(config.data.data_path.resolve()),
        "inventory_sha256": digest,
        "file_count": len(files),
    }


def collect_environment(distributed: DistributedContext) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    gpu = None
    if distributed.device.type == "cuda":
        gpu = torch.cuda.get_device_name(distributed.device)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "world_size": distributed.world_size,
        "git_commit": git_commit,
    }


def _validate_run_name(run_name: str) -> None:
    if not run_name or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name) is None:
        raise ValueError(
            "run_name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens."
        )


def create_run_directory(
    output_root: Path,
    distributed: DistributedContext,
    *,
    requested_name: str | None,
    lora_rank: int,
    stride: int,
    finetune_mode: str = "lora",
) -> Path:
    """Create one non-overwriting run directory and share its name across ranks."""
    payload: list[dict[str, str] | None] = [None]
    if distributed.is_main:
        try:
            runs_dir = output_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            if requested_name is not None:
                _validate_run_name(requested_name)
                run_name = requested_name
                run_dir = runs_dir / run_name
                if run_dir.exists():
                    raise FileExistsError(
                        f"Run directory already exists and will not be overwritten: {run_dir}"
                    )
            else:
                capacity = f"r{lora_rank}" if finetune_mode == "lora" else "full"
                base_name = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                    f"{capacity}_s{stride}"
                )
                run_name = base_name
                suffix = 1
                while (runs_dir / run_name).exists():
                    run_name = f"{base_name}_{suffix:02d}"
                    suffix += 1
                run_dir = runs_dir / run_name
            run_dir.mkdir(parents=False, exist_ok=False)
            (run_dir / "plots").mkdir()
            (run_dir / "resume").mkdir()
            payload[0] = {"run_name": run_name}
        except Exception as exc:
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}

    if distributed.enabled:
        dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if result is None:
        raise RuntimeError("Failed to resolve the experiment run directory.")
    if "error" in result:
        raise RuntimeError(result["error"])
    run_dir = output_root / "runs" / result["run_name"]
    if distributed.enabled:
        dist.barrier()
    return run_dir


def resolve_resume_checkpoint(
    requested_path: Path,
    output_root: Path,
    distributed: DistributedContext,
    *,
    requested_run_name: str | None,
) -> tuple[Path, Path]:
    """Resolve a latest checkpoint on rank 0 and broadcast it to every rank."""
    payload: list[dict[str, str] | None] = [None]
    if distributed.is_main:
        try:
            path = requested_path.expanduser().resolve()
            if path.is_file():
                if path.name != "latest.json":
                    raise ValueError(f"Unsupported resume file: {path}")
                latest_path = path
                resume_dir = path.parent
            elif (path / "trainer_state.json").is_file():
                resume_dir = path.parent
                latest_path = resume_dir / "latest.json"
            elif (path / "resume" / "latest.json").is_file():
                resume_dir = path / "resume"
                latest_path = resume_dir / "latest.json"
            elif (path / "latest.json").is_file():
                resume_dir = path
                latest_path = path / "latest.json"
            else:
                raise FileNotFoundError(
                    "Could not find trainer_state.json or resume/latest.json under "
                    f"{path}."
                )

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            checkpoint_dir = (resume_dir / str(latest["checkpoint"])).resolve()
            if checkpoint_dir.parent != resume_dir.resolve():
                raise ValueError(
                    f"latest.json points outside its resume directory: {checkpoint_dir}"
                )
            if (path / "trainer_state.json").is_file() and path != checkpoint_dir:
                raise ValueError(
                    "Only the checkpoint referenced by resume/latest.json can be "
                    "resumed in-place; use the latest checkpoint to avoid rewriting "
                    "newer experiment history."
                )
            if not (checkpoint_dir / "trainer_state.json").is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint is incomplete: {checkpoint_dir}"
                )
            run_dir = resume_dir.parent
            if (run_dir / "result.json").is_file():
                raise ValueError(
                    f"Run is already complete and cannot be resumed in-place: {run_dir}"
                )
            if requested_run_name is not None and requested_run_name != run_dir.name:
                raise ValueError(
                    f"--run-name {requested_run_name!r} does not match resumed run "
                    f"{run_dir.name!r}."
                )
            payload[0] = {
                "checkpoint_dir": str(checkpoint_dir),
                "run_dir": str(run_dir),
            }
        except Exception as exc:
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed.enabled:
        dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if result is None:
        raise RuntimeError("Failed to resolve resume checkpoint.")
    if "error" in result:
        raise RuntimeError(result["error"])
    checkpoint_dir = Path(result["checkpoint_dir"])
    run_dir = Path(result["run_dir"])
    if not run_dir.is_relative_to(output_root.resolve()):
        raise ValueError(
            f"Resume run must be inside configured output root {output_root.resolve()}."
        )
    return checkpoint_dir, run_dir


def write_loss_svg(path: Path, records: list[dict[str, float | int | str]]) -> None:
    """Render train/validation loss without adding a plotting dependency."""
    series = {
        "train": [
            (float(item["global_step"]), float(item["loss"]))
            for item in records
            if item["split"] == "train"
        ],
        "validation": [
            (float(item["global_step"]), float(item["loss"]))
            for item in records
            if item["split"] == "validation"
        ],
    }
    points = series["train"] + series["validation"]
    if not points:
        return
    width, height = 960, 540
    left, right, top, bottom = 80, 30, 35, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max = max(1.0, max(point[0] for point in points))
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)
    if y_max == y_min:
        y_max = y_min + 1.0
    margin = (y_max - y_min) * 0.08
    y_min -= margin
    y_max += margin

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + x / x_max * plot_width
        py = top + (y_max - y) / (y_max - y_min) * plot_height
        return px, py

    train_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in map(transform, series["train"])
    )
    validation_circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#d1495b"/>'
        for x, y in map(transform, series["validation"])
    )
    train_polyline = (
        f'<polyline points="{train_points}" fill="none" stroke="#277da1" '
        'stroke-width="2"/>'
        if train_points
        else ""
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Training and validation loss</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif">optimizer step</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="sans-serif">loss</text>
<text x="{left}" y="{height - bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">0</text>
<text x="{width - right}" y="{height - bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(x_max)}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.4f}</text>
<text x="{left - 8}" y="{height - bottom}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.4f}</text>
{train_polyline}
{validation_circles}
<line x1="{width - 210}" y1="42" x2="{width - 180}" y2="42" stroke="#277da1" stroke-width="2"/>
<text x="{width - 172}" y="46" font-family="sans-serif" font-size="12">train</text>
<circle cx="{width - 110}" cy="42" r="5" fill="#d1495b"/>
<text x="{width - 100}" y="46" font-family="sans-serif" font-size="12">validation</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".svg.tmp")
    temporary.write_text(svg, encoding="utf-8")
    temporary.replace(path)


def write_training_plots(
    plots_dir: Path,
    records: list[dict[str, float | int | str]],
) -> None:
    """Write headless PNG loss and learning-rate plots atomically."""
    if not records:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG plots require matplotlib. Install finetune_lora/requirements.txt."
        ) from exc

    plots_dir.mkdir(parents=True, exist_ok=True)
    train = [item for item in records if item["split"] == "train"]
    validation = [item for item in records if item["split"] == "validation"]

    def save_figure(filename: str) -> None:
        destination = plots_dir / filename
        temporary = destination.with_name(f".{destination.name}.tmp")
        plt.tight_layout()
        plt.savefig(temporary, format="png", dpi=150)
        plt.close()
        temporary.replace(destination)

    if train or validation:
        plt.figure(figsize=(9.6, 5.4))
        if train:
            plt.plot(
                [item["global_step"] for item in train],
                [item["loss"] for item in train],
                label="train",
            )
        if validation:
            plt.scatter(
                [item["global_step"] for item in validation],
                [item["loss"] for item in validation],
                label="validation selection",
                color="#d1495b",
            )
            mixed_points = [
                item for item in validation if "mixed_loss" in item
            ]
            if mixed_points:
                plt.plot(
                    [item["global_step"] for item in mixed_points],
                    [item["mixed_loss"] for item in mixed_points],
                    label="validation mixed objective",
                    color="#9c6644",
                    linestyle="--",
                )
        plt.xlabel("optimizer step")
        plt.ylabel("loss")
        plt.title("Training and validation loss")
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure("loss.png")

    component_series = (
        (train, "s1_loss", "train s1", "#277da1", "-"),
        (train, "s2_loss", "train s2", "#43aa8b", "-"),
        (validation, "s1_loss", "validation s1", "#d1495b", "--"),
        (validation, "s2_loss", "validation s2", "#f8961e", "--"),
    )
    if any(any(key in item for item in items) for items, key, *_ in component_series):
        plt.figure(figsize=(9.6, 5.4))
        for items, key, label, color, linestyle in component_series:
            points = [item for item in items if key in item]
            if not points:
                continue
            plt.plot(
                [item["global_step"] for item in points],
                [item[key] for item in points],
                label=label,
                color=color,
                linestyle=linestyle,
            )
        plt.xlabel("optimizer step")
        plt.ylabel("cross-entropy loss")
        plt.title("s1 and s2 component losses")
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure("component_loss.png")

    conditioning_series = (
        ("predicted_s2_loss", "predicted s2"),
        ("teacher_s2_loss", "teacher s2"),
        ("mixed_s2_loss", "mixed training objective"),
    )
    if any(any(key in item for item in validation) for key, _ in conditioning_series):
        plt.figure(figsize=(9.6, 5.4))
        for key, label in conditioning_series:
            points = [item for item in validation if key in item]
            if points:
                plt.plot(
                    [item["global_step"] for item in points],
                    [item[key] for item in points],
                    label=label,
                )
        plt.xlabel("optimizer step")
        plt.ylabel("cross-entropy loss")
        plt.title("Validation s2 conditioning losses")
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure("conditioning_loss.png")

    accuracy_series = (
        ("s1_top1_accuracy", "s1 top-1"),
        ("s1_top5_accuracy", "s1 top-5"),
        ("s2_top1_accuracy", "s2 top-1"),
        ("s2_top5_accuracy", "s2 top-5"),
        ("teacher_s2_top1_accuracy", "teacher s2 top-1"),
        ("teacher_s2_top5_accuracy", "teacher s2 top-5"),
    )
    if any(any(key in item for item in validation) for key, _ in accuracy_series):
        plt.figure(figsize=(9.6, 5.4))
        for key, label in accuracy_series:
            points = [item for item in validation if key in item]
            if points:
                plt.plot(
                    [item["global_step"] for item in points],
                    [item[key] for item in points],
                    label=label,
                )
        plt.xlabel("optimizer step")
        plt.ylabel("accuracy")
        plt.title("Validation token top-k accuracy")
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure("token_accuracy.png")

    entropy_series = (
        ("s1_mean_entropy", "s1"),
        ("s2_mean_entropy", "s2"),
        ("teacher_s2_mean_entropy", "teacher s2"),
    )
    if any(any(key in item for item in validation) for key, _ in entropy_series):
        plt.figure(figsize=(9.6, 5.4))
        for key, label in entropy_series:
            points = [item for item in validation if key in item]
            if points:
                plt.plot(
                    [item["global_step"] for item in points],
                    [item[key] for item in points],
                    label=label,
                )
        plt.xlabel("optimizer step")
        plt.ylabel("entropy (nats)")
        plt.title("Validation token mean entropy")
        plt.grid(alpha=0.25)
        plt.legend()
        save_figure("token_entropy.png")

    learning_rate = [
        item
        for item in records
        if item["split"] != "test" and "learning_rate" in item
    ]
    if learning_rate:
        plt.figure(figsize=(9.6, 5.4))
        plt.plot(
            [item["global_step"] for item in learning_rate],
            [item["learning_rate"] for item in learning_rate],
            color="#277da1",
        )
        plt.xlabel("optimizer step")
        plt.ylabel("learning rate")
        plt.title("Learning-rate schedule")
        plt.grid(alpha=0.25)
        save_figure("learning_rate.png")

    gradient_records = [
        item for item in records if item["split"] == "gradient"
    ]
    if gradient_records:
        plt.figure(figsize=(9.6, 5.4))
        gradient_series = (
            ("grad_norm_total", "total"),
            ("grad_norm_proj_s1", "proj_s1"),
            ("grad_norm_transformer", "transformer"),
            ("grad_norm_dependency", "dependency"),
        )
        plotted = False
        for key, label in gradient_series:
            points = [
                item for item in gradient_records if float(item.get(key, 0.0)) > 0
            ]
            if not points:
                continue
            plotted = True
            plt.plot(
                [item["global_step"] for item in points],
                [item[key] for item in points],
                label=label,
            )
        if plotted:
            plt.xlabel("optimizer step")
            plt.ylabel("pre-clipping gradient L2 norm")
            plt.title("LoRA gradient norms")
            plt.yscale("log")
            plt.grid(alpha=0.25)
            plt.legend()
            save_figure("gradient_norms.png")
        else:
            plt.close()

    conflict_records = [
        item for item in records if item["split"] == "gradient_conflict"
    ]
    if conflict_records:
        steps = [item["global_step"] for item in conflict_records]
        figure, axes = plt.subplots(2, 1, figsize=(9.6, 7.2), sharex=True)
        axes[0].plot(
            steps,
            [item["transformer_s1_grad_norm"] for item in conflict_records],
            label="s1",
            color="#277da1",
        )
        axes[0].plot(
            steps,
            [item["transformer_s2_grad_norm"] for item in conflict_records],
            label="s2",
            color="#f8961e",
        )
        axes[0].set_ylabel("gradient L2 norm")
        axes[0].set_title("Shared Transformer task gradients")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(
            steps,
            [item["transformer_grad_cosine"] for item in conflict_records],
            color="#7b2cbf",
        )
        axes[1].axhline(0.0, color="#333333", linewidth=1, linestyle="--")
        axes[1].axhline(-0.1, color="#d1495b", linewidth=1, linestyle=":")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("cosine similarity")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].grid(alpha=0.25)
        save_figure("gradient_conflict.png")


def summarize_gradient_conflicts(
    records: list[dict[str, float | int | str]],
) -> dict[str, float | int] | None:
    """Summarize sampled shared-gradient directions for final reporting."""
    conflict_records = [
        item for item in records if item["split"] == "gradient_conflict"
    ]
    if not conflict_records:
        return None
    cosines = np.asarray(
        [item["transformer_grad_cosine"] for item in conflict_records],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [item["transformer_s2_to_s1_ratio"] for item in conflict_records],
        dtype=np.float64,
    )
    return {
        "sample_count": int(cosines.size),
        "median_cosine": float(np.median(cosines)),
        "negative_cosine_ratio": float(np.mean(cosines < 0.0)),
        "strong_conflict_ratio": float(np.mean(cosines < -0.1)),
        "median_s2_to_s1_ratio": float(np.median(ratios)),
    }


def _capture_rank_state(
    device: torch.device,
    *,
    running_loss: float,
    running_s1_loss: float,
    running_s2_loss: float,
    running_batches: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "running_loss": running_loss,
        "running_s1_loss": running_s1_loss,
        "running_s2_loss": running_s2_loss,
        "running_batches": running_batches,
    }
    if device.type == "cuda":
        state["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rank_state(
    path: Path, device: torch.device
) -> tuple[float, float, float, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    if "running_s1_loss" not in state or "running_s2_loss" not in state:
        # Old checkpoints cannot reconstruct component accumulators. Reset only
        # the incomplete logging window; model/optimizer resume is unaffected.
        return 0.0, 0.0, 0.0, 0
    return (
        float(state["running_loss"]),
        float(state["running_s1_loss"]),
        float(state["running_s2_loss"]),
        int(state["running_batches"]),
    )


def _resume_signature(
    config: ExperimentConfig,
    distributed: DistributedContext,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "finetune_mode": config.training.finetune_mode,
        "data_fingerprint": fingerprint,
        "context_length": config.data.context_length,
        "stride": config.data.stride,
        "target_mode": config.data.target_mode,
        "target_length": config.data.target_length,
        "clip": config.data.clip,
        "batch_size_per_rank": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "validation_interval": config.training.validation_interval,
        "s1_loss_weight": config.training.s1_loss_weight,
        "s2_loss_weight": config.training.s2_loss_weight,
        "s2_conditioning": config.training.s2_conditioning,
        "s2_teacher_loss_weight": config.training.s2_teacher_loss_weight,
        "s2_predicted_loss_weight": config.training.s2_predicted_loss_weight,
        "s2_top_k": config.training.s2_top_k,
        "s2_temperature": config.training.s2_temperature,
        "early_stopping_min_delta": config.training.early_stopping_min_delta,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "adam_beta1": config.training.adam_beta1,
        "adam_beta2": config.training.adam_beta2,
        "max_grad_norm": config.training.max_grad_norm,
        "seed": config.training.seed,
        "model_display_name": config.training.model_display_name,
        "world_size": distributed.world_size,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "proj_s1_rank": config.lora.proj_s1_rank,
        "proj_s1_alpha": config.lora.proj_s1_alpha,
        "full_finetune_proj_s1": config.lora.full_finetune_proj_s1,
        "dependency_rank": config.lora.dependency_rank,
        "dependency_alpha": config.lora.dependency_alpha,
        "fusion_rank": config.lora.fusion_rank,
        "fusion_alpha": config.lora.fusion_alpha,
        "ffn_rank": config.lora.ffn_rank,
        "ffn_alpha": config.lora.ffn_alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": list(config.lora.target_modules),
        "exclude_modules": list(config.lora.exclude_modules),
        "bias": config.lora.bias,
        "base_model_path": str(config.lora.base_model_path.resolve()),
        "tokenizer_path": str(config.lora.tokenizer_path.resolve()),
    }


def save_resume_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    run_dir: Path,
    distributed: DistributedContext,
    *,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    best_validation_loss: float,
    early_stopping_reference_loss: float,
    stale_epochs: int,
    validation_improved_this_epoch: bool,
    history_count: int,
    metric_count: int,
    running_loss: float,
    running_s1_loss: float,
    running_s2_loss: float,
    running_batches: int,
    signature: dict[str, Any],
) -> Path:
    """Atomically save a complete optimizer-boundary resume checkpoint."""
    resume_dir = run_dir / "resume"
    checkpoint_name = f"step-{global_step:08d}"
    checkpoint_dir = resume_dir / checkpoint_name
    temporary_dir = resume_dir / f".tmp-{checkpoint_name}"
    status: list[dict[str, str] | None] = [None]
    if distributed.is_main:
        try:
            if checkpoint_dir.exists() or temporary_dir.exists():
                raise FileExistsError(
                    f"Resume checkpoint already exists: {checkpoint_dir}"
                )
            temporary_dir.mkdir(parents=True)
            model_to_save = model.module if distributed.enabled else model
            model_to_save.save_pretrained(temporary_dir)
            torch.save(optimizer.state_dict(), temporary_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), temporary_dir / "scheduler.pt")
            save_json(
                temporary_dir / "trainer_state.json",
                {
                    "epoch": epoch,
                    "next_batch_index": next_batch_index,
                    "global_step": global_step,
                    "best_validation_loss": best_validation_loss,
                    "early_stopping_reference_loss": (
                        early_stopping_reference_loss
                    ),
                    "stale_epochs": stale_epochs,
                    "validation_improved_this_epoch": (
                        validation_improved_this_epoch
                    ),
                    "history_count": history_count,
                    "metric_count": metric_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "resume_signature": signature,
                },
            )
            status[0] = {"ok": "true"}
        except Exception as exc:
            status[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed.enabled:
        dist.broadcast_object_list(status, src=0)
    if status[0] is None or "error" in status[0]:
        message = "Unknown checkpoint error" if status[0] is None else status[0]["error"]
        raise RuntimeError(message)

    torch.save(
        _capture_rank_state(
            distributed.device,
            running_loss=running_loss,
            running_s1_loss=running_s1_loss,
            running_s2_loss=running_s2_loss,
            running_batches=running_batches,
        ),
        temporary_dir / f"rng_state_rank-{distributed.rank:04d}.pt",
    )
    if distributed.enabled:
        dist.barrier()
    if distributed.is_main:
        temporary_dir.replace(checkpoint_dir)
        save_json(
            resume_dir / "latest.json",
            {"checkpoint": checkpoint_name, "global_step": global_step},
        )
        print(f"Saved resumable checkpoint to {checkpoint_dir}")
    if distributed.enabled:
        dist.barrier()
    return checkpoint_dir


def load_resume_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler,
    distributed: DistributedContext,
    *,
    expected_signature: dict[str, Any],
) -> tuple[dict[str, Any], float, float, float, int]:
    state = json.loads(
        (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    saved_signature = state.get("resume_signature")
    if (
        isinstance(saved_signature, dict)
        and "exclude_modules" in expected_signature
        and "exclude_modules" not in saved_signature
    ):
        # Older Q/V checkpoints did not support exclusions.
        saved_signature = {**saved_signature, "exclude_modules": []}
    if (
        isinstance(saved_signature, dict)
        and "validation_interval" in expected_signature
        and "validation_interval" not in saved_signature
    ):
        # Checkpoints written before step-level validation existed used only
        # epoch-end validation, which is equivalent to interval zero.
        saved_signature = {**saved_signature, "validation_interval": 0}
    if isinstance(saved_signature, dict):
        # The original objective used equal s1/s2 weighting and treated every
        # strict validation decrease as an early-stopping improvement.
        legacy_defaults = {
            "finetune_mode": "lora",
            "s1_loss_weight": 0.5,
            "s2_loss_weight": 0.5,
            "s2_conditioning": "teacher",
            "s2_teacher_loss_weight": 0.1,
            "s2_predicted_loss_weight": 0.9,
            "s2_top_k": 5,
            "s2_temperature": 1.0,
            "early_stopping_min_delta": 0.0,
            "warmup_steps": 0,
            "model_display_name": None,
            "proj_s1_rank": None,
            "proj_s1_alpha": None,
            "full_finetune_proj_s1": False,
            "dependency_rank": None,
            "dependency_alpha": None,
            "fusion_rank": None,
            "fusion_alpha": None,
            "ffn_rank": None,
            "ffn_alpha": None,
            "target_mode": "all",
            "target_length": None,
        }
        if "tokenizer_path" in expected_signature:
            # Older checkpoints did not record this path. Treat their tokenizer
            # as the one explicitly selected for the resumed run.
            legacy_defaults["tokenizer_path"] = expected_signature["tokenizer_path"]
        for key, value in legacy_defaults.items():
            if key in expected_signature and key not in saved_signature:
                saved_signature = {**saved_signature, key: value}
    if saved_signature != expected_signature:
        raise ValueError(
            "Resume configuration or dataset does not match the checkpoint. "
            "Keep the same data cache, batch/accumulation settings, world size, "
            "validation interval, seed, and LoRA configuration."
        )
    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt",
        map_location=distributed.device,
        weights_only=False,
    )
    scheduler_state = torch.load(
        checkpoint_dir / "scheduler.pt",
        map_location=distributed.device,
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    rank_state = checkpoint_dir / f"rng_state_rank-{distributed.rank:04d}.pt"
    if not rank_state.is_file():
        raise FileNotFoundError(f"Missing per-rank resume state: {rank_state}")
    running_loss, running_s1_loss, running_s2_loss, running_batches = (
        _restore_rank_state(rank_state, distributed.device)
    )
    return (
        state,
        running_loss,
        running_s1_loss,
        running_s2_loss,
        running_batches,
    )


def train(config: ExperimentConfig) -> list[dict[str, float]]:
    """Run single-device or torchrun LoRA/full-parameter training."""
    config.validate()
    distributed = initialize_distributed(config.training.device)
    try:
        return _train(config, distributed)
    finally:
        cleanup_distributed(distributed)


def _train(
    config: ExperimentConfig, distributed: DistributedContext
) -> list[dict[str, float]]:
    if distributed.enabled and config.data.token_cache_path is None:
        raise ValueError(
            "DDP training requires --token-cache-path so each rank does not "
            "load the complete raw Parquet dataset."
        )
    set_seed(config.training.seed + distributed.rank)
    device = distributed.device
    precision = resolve_precision(config.training.precision, device)
    torch.set_float32_matmul_precision("high")
    resume_checkpoint: Path | None = None
    initialization_source: Path | None = None
    if config.training.resume_from is not None:
        resume_checkpoint, run_dir = resolve_resume_checkpoint(
            config.training.resume_from,
            config.lora.adapter_output_dir,
            distributed,
            requested_run_name=config.training.run_name,
        )
    else:
        run_dir = create_run_directory(
            config.lora.adapter_output_dir,
            distributed,
            requested_name=config.training.run_name,
            lora_rank=config.lora.rank,
            stride=config.data.stride,
            finetune_mode=config.training.finetune_mode,
        )
    if distributed.is_main:
        print(
            f"device={device} precision={precision} "
            f"world_size={distributed.world_size} "
            f"finetune_mode={config.training.finetune_mode}"
        )
        print(
            "training_and_selection_s2_conditioning="
            f"{config.training.s2_conditioning}"
        )
        print(
            f"target_mode={config.data.target_mode} "
            f"target_length={config.data.target_length}"
        )
        print(
            "full_finetune_proj_s1="
            f"{config.lora.full_finetune_proj_s1}"
        )
        if config.training.s2_conditioning.startswith("mixed-"):
            print(
                "mixed_s2_branch_weights="
                f"teacher:{config.training.s2_teacher_loss_weight:g},"
                f"predicted:{config.training.s2_predicted_loss_weight:g}"
            )
        if "topk" in config.training.s2_conditioning:
            print(
                f"s2_top_k={config.training.s2_top_k} "
                f"s2_temperature={config.training.s2_temperature:g}"
            )
        print(f"run_directory={run_dir}")
    loaders, datasets, boundaries = create_loaders(config, distributed)
    fingerprint = dataset_fingerprint(config)
    signature = _resume_signature(config, distributed, fingerprint)

    tokenizer = None
    if config.data.token_cache_path is None:
        tokenizer = load_frozen_tokenizer(config.lora).to(device)
    if resume_checkpoint is not None and config.training.finetune_mode == "full":
        model = load_full_finetune_model(resume_checkpoint)
        model.requires_grad_(True)
        matched_modules = ["all Kronos predictor parameters"]
    elif resume_checkpoint is not None:
        model = load_qv_lora_adapter(
            config.lora,
            resume_checkpoint,
            is_trainable=True,
        )
        matched_modules = list(config.lora.target_modules)
    elif config.training.initialize_from_adapter is not None:
        (
            model,
            matched_modules,
            initialization_source,
            initialized_tensor_count,
        ) = initialize_qv_lora_model(
            config.lora,
            config.training.initialize_from_adapter,
            model_selection=config.training.initialize_model_selection,
        )
        if distributed.is_main:
            print(
                f"initialized_adapter={initialization_source} "
                f"copied_tensors={initialized_tensor_count}"
            )
    elif config.training.finetune_mode == "full":
        model = build_full_finetune_model(config.lora)
        matched_modules = ["all Kronos predictor parameters"]
    else:
        model, matched_modules = build_qv_lora_model(config.lora)
    model = model.to(device)
    if distributed.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if distributed.is_main:
        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"trainable_parameters={trainable_count:,} "
            f"total_parameters={total_count:,} "
            f"ratio={trainable_count / total_count:.4%}"
        )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loaders["train"]) / config.training.gradient_accumulation_steps
    )
    total_optimizer_steps = updates_per_epoch * config.training.epochs
    scheduler = build_learning_rate_scheduler(
        optimizer,
        total_steps=max(1, total_optimizer_steps),
        warmup_steps=config.training.warmup_steps,
    )
    if distributed.is_main:
        schedule_name = (
            "linear_warmup_cosine"
            if config.training.warmup_steps > 0
            else "cosine"
        )
        print(
            f"lr_schedule={schedule_name} "
            f"warmup_steps={config.training.warmup_steps} "
            f"total_optimizer_steps={total_optimizer_steps}"
        )

    best_dir = run_dir / "best_model"
    last_dir = run_dir / "last_model"
    metrics_path = run_dir / "metrics.jsonl"
    loss_plot_path = run_dir / "plots" / "loss.svg"
    if distributed.is_main and resume_checkpoint is None:
        saved_config = config.to_dict()
        saved_config["resolved_run_name"] = run_dir.name
        saved_config["run_directory"] = str(run_dir)
        save_json(run_dir / "config.json", saved_config)
        save_json(
            run_dir / "dataset_summary.json",
            {
                "train_end": str(boundaries.train_end),
                "validation_end": str(boundaries.validation_end),
                "target_mode": config.data.target_mode,
                "target_length": config.data.target_length,
                "windows": {name: len(dataset) for name, dataset in datasets.items()},
                "matched_trainable_modules": matched_modules,
                # Retain the old field for analysis scripts consuming LoRA runs.
                "matched_lora_modules": (
                    matched_modules
                    if config.training.finetune_mode == "lora"
                    else []
                ),
                "resolved_precision": precision,
                "world_size": distributed.world_size,
                "data_fingerprint": fingerprint,
            },
        )
        save_json(run_dir / "environment.json", collect_environment(distributed))

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    early_stopping_reference_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    metric_records: list[dict[str, float | int | str]] = []
    start_epoch = 1
    resume_next_batch = 1
    resumed_running_loss = 0.0
    resumed_running_s1_loss = 0.0
    resumed_running_s2_loss = 0.0
    resumed_running_batches = 0
    resumed_validation_improved_this_epoch = False

    if resume_checkpoint is not None:
        (
            resume_state,
            resumed_running_loss,
            resumed_running_s1_loss,
            resumed_running_s2_loss,
            resumed_running_batches,
        ) = load_resume_state(
            resume_checkpoint,
            optimizer,
            scheduler,
            distributed,
            expected_signature=signature,
        )
        start_epoch = int(resume_state["epoch"])
        resume_next_batch = int(resume_state["next_batch_index"])
        global_step = int(resume_state["global_step"])
        best_validation_loss = float(resume_state["best_validation_loss"])
        early_stopping_reference_loss = float(
            resume_state.get(
                "early_stopping_reference_loss", best_validation_loss
            )
        )
        stale_epochs = int(resume_state["stale_epochs"])
        resumed_validation_improved_this_epoch = bool(
            resume_state.get("validation_improved_this_epoch", False)
        )
        history_count = int(resume_state["history_count"])
        metric_count = int(resume_state["metric_count"])
        history_path = run_dir / "history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
        history = history[:history_count]
        metric_records = load_jsonl(metrics_path)[:metric_count]
        if start_epoch > config.training.epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} exceeds configured epochs "
                f"{config.training.epochs}."
            )
        if distributed.is_main:
            save_json(run_dir / "history.json", history)
            rewrite_jsonl(metrics_path, metric_records)
            write_loss_svg(loss_plot_path, metric_records)
            write_training_plots(run_dir / "plots", metric_records)
            print(
                f"Resumed {run_dir.name} at epoch={start_epoch} "
                f"next_batch={resume_next_batch} step={global_step}"
            )
        if distributed.enabled:
            dist.barrier()
        # Plotting and history repair happen after the first restore. Restore
        # once more immediately before training so auxiliary work cannot alter
        # Python/NumPy/PyTorch random streams.
        (
            resumed_running_loss,
            resumed_running_s1_loss,
            resumed_running_s2_loss,
            resumed_running_batches,
        ) = _restore_rank_state(
            resume_checkpoint / f"rng_state_rank-{distributed.rank:04d}.pt",
            device,
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_sampler = loaders["train"].sampler
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if resume_checkpoint is not None and epoch == start_epoch:
            running_loss = resumed_running_loss
            running_s1_loss = resumed_running_s1_loss
            running_s2_loss = resumed_running_s2_loss
            running_batches = resumed_running_batches
            validation_improved_this_epoch = (
                resumed_validation_improved_this_epoch
            )
        else:
            running_loss = 0.0
            running_s1_loss = 0.0
            running_s2_loss = 0.0
            running_batches = 0
            validation_improved_this_epoch = False
        conflict_probe_batches: list[Any] = []

        for batch_index, batch in enumerate(loaders["train"], start=1):
            if (
                resume_checkpoint is not None
                and epoch == start_epoch
                and batch_index < resume_next_batch
            ):
                continue
            group_start = (
                (batch_index - 1)
                // config.training.gradient_accumulation_steps
                * config.training.gradient_accumulation_steps
                + 1
            )
            group_end = min(
                group_start + config.training.gradient_accumulation_steps - 1,
                len(loaders["train"]),
            )
            actual_group_size = group_end - group_start + 1
            should_update = (
                batch_index % config.training.gradient_accumulation_steps == 0
                or batch_index == len(loaders["train"])
            )
            should_probe_group = (
                config.training.gradient_conflict_interval > 0
                and (global_step + 1)
                % config.training.gradient_conflict_interval
                == 0
            )
            if should_probe_group:
                conflict_probe_batches.append(batch)
            sync_context = (
                model.no_sync()
                if distributed.enabled and not should_update
                else nullcontext()
            )
            with sync_context:
                metrics = forward_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    precision=precision,
                    s1_loss_weight=config.training.s1_loss_weight,
                    s2_loss_weight=config.training.s2_loss_weight,
                    s2_conditioning=config.training.s2_conditioning,
                    s2_teacher_loss_weight=(
                        config.training.s2_teacher_loss_weight
                    ),
                    s2_predicted_loss_weight=(
                        config.training.s2_predicted_loss_weight
                    ),
                    s2_top_k=config.training.s2_top_k,
                    s2_temperature=config.training.s2_temperature,
                )
                scaled_loss = metrics.loss / actual_group_size
                scaled_loss.backward()
            running_loss += metrics.loss.item()
            running_s1_loss += metrics.s1_loss.item()
            running_s2_loss += metrics.s2_loss.item()
            running_batches += 1

            if should_update:
                next_global_step = global_step + 1
                should_log_gradients = (
                    config.training.gradient_log_interval > 0
                    and next_global_step
                    % config.training.gradient_log_interval
                    == 0
                )
                gradient_metrics = (
                    lora_gradient_norms(model) if should_log_gradients else None
                )
                pre_clip_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters, config.training.max_grad_norm
                )
                if gradient_metrics is not None:
                    pre_clip_value = float(pre_clip_norm.item())
                    gradient_metrics["grad_norm_total"] = pre_clip_value
                    gradient_metrics["grad_clip_scale"] = min(
                        1.0,
                        config.training.max_grad_norm
                        / (pre_clip_value + 1e-12),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if should_probe_group:
                    conflict_metrics = task_gradient_conflict_metrics(
                        model,
                        tokenizer,
                        conflict_probe_batches,
                        device,
                        precision=precision,
                        distributed=distributed,
                        s1_weight=config.training.s1_loss_weight,
                        s2_weight=config.training.s2_loss_weight,
                        s2_conditioning=config.training.s2_conditioning,
                        s2_teacher_loss_weight=(
                            config.training.s2_teacher_loss_weight
                        ),
                        s2_predicted_loss_weight=(
                            config.training.s2_predicted_loss_weight
                        ),
                        s2_top_k=config.training.s2_top_k,
                        s2_temperature=config.training.s2_temperature,
                    )
                    conflict_probe_batches.clear()
                    if distributed.is_main:
                        conflict_record: dict[str, float | int | str] = {
                            "split": "gradient_conflict",
                            "epoch": epoch,
                            "global_step": global_step,
                            **conflict_metrics,
                        }
                        metric_records.append(conflict_record)
                        append_jsonl(metrics_path, conflict_record)
                        write_training_plots(run_dir / "plots", metric_records)
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"shared_g1="
                            f"{conflict_metrics['transformer_s1_grad_norm']:.3e} "
                            f"shared_g2="
                            f"{conflict_metrics['transformer_s2_grad_norm']:.3e} "
                            f"ratio="
                            f"{conflict_metrics['transformer_s2_to_s1_ratio']:.3f} "
                            f"effective_ratio="
                            f"{conflict_metrics['effective_s2_to_s1_ratio']:.3f} "
                            f"cos="
                            f"{conflict_metrics['transformer_grad_cosine']:.4f}"
                        )

                if gradient_metrics is not None and distributed.is_main:
                    gradient_record: dict[str, float | int | str] = {
                        "split": "gradient",
                        "epoch": epoch,
                        "global_step": global_step,
                        **gradient_metrics,
                    }
                    metric_records.append(gradient_record)
                    append_jsonl(metrics_path, gradient_record)
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"grad_total={gradient_metrics['grad_norm_total']:.3e} "
                        f"grad_s1={gradient_metrics['grad_norm_proj_s1']:.3e} "
                        f"grad_transformer="
                        f"{gradient_metrics['grad_norm_transformer']:.3e} "
                        f"grad_dependency="
                        f"{gradient_metrics['grad_norm_dependency']:.3e} "
                        f"clip_scale={gradient_metrics['grad_clip_scale']:.3f}"
                    )

                if global_step % config.training.log_interval == 0:
                    log_totals = torch.tensor(
                        [
                            running_loss,
                            running_s1_loss,
                            running_s2_loss,
                            running_batches,
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    if distributed.enabled:
                        dist.all_reduce(log_totals, op=dist.ReduceOp.SUM)
                    if distributed.is_main:
                        denominator = log_totals[3].item()
                        train_loss = log_totals[0].item() / denominator
                        train_s1_loss = log_totals[1].item() / denominator
                        train_s2_loss = log_totals[2].item() / denominator
                        metric_record: dict[str, float | int | str] = {
                            "split": "train",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": train_loss,
                            "s1_loss": train_s1_loss,
                            "s2_loss": train_s2_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        metric_records.append(metric_record)
                        append_jsonl(metrics_path, metric_record)
                        write_loss_svg(loss_plot_path, metric_records)
                        write_training_plots(run_dir / "plots", metric_records)
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={train_loss:.6f} "
                            f"s1={train_s1_loss:.6f} "
                            f"s2={train_s2_loss:.6f} "
                            f"lr={optimizer.param_groups[0]['lr']:.3e}"
                        )
                    running_loss = 0.0
                    running_s1_loss = 0.0
                    running_s2_loss = 0.0
                    running_batches = 0

                if should_run_step_validation(
                    global_step,
                    config.training.validation_interval,
                    is_epoch_end=batch_index == len(loaders["train"]),
                ):
                    validation = evaluate(
                        model,
                        tokenizer,
                        loaders["validation"],
                        device,
                        split="validation",
                        epoch=epoch,
                        precision=precision,
                        s1_loss_weight=config.training.s1_loss_weight,
                        s2_loss_weight=config.training.s2_loss_weight,
                        s2_conditioning=config.training.s2_conditioning,
                        s2_teacher_loss_weight=(
                            config.training.s2_teacher_loss_weight
                        ),
                        s2_predicted_loss_weight=(
                            config.training.s2_predicted_loss_weight
                        ),
                        s2_top_k=config.training.s2_top_k,
                        s2_temperature=config.training.s2_temperature,
                        distributed=distributed,
                    )
                    if distributed.is_main:
                        validation_record: dict[str, float | int | str] = {
                            "split": "validation",
                            "validation_scope": "step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": validation["loss"],
                            "s1_loss": validation["s1_loss"],
                            "s2_loss": validation["s2_loss"],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            **{
                                key: validation[key]
                                for key in PREDICTION_METRIC_KEYS
                            },
                            **{
                                key: validation[key]
                                for key in MIXED_CONDITIONING_METRIC_KEYS
                                if key in validation
                            },
                        }
                        metric_records.append(validation_record)
                        append_jsonl(metrics_path, validation_record)
                        write_loss_svg(loss_plot_path, metric_records)
                        write_training_plots(run_dir / "plots", metric_records)
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"validation_loss={validation['loss']:.6f} "
                            f"s1={validation['s1_loss']:.6f} "
                            f"s2={validation['s2_loss']:.6f}"
                        )
                        print(
                            f"validation_tokens step={global_step} "
                            f"s1_top1={validation['s1_top1_accuracy']:.4%} "
                            f"s1_top5={validation['s1_top5_accuracy']:.4%} "
                            f"s2_top1={validation['s2_top1_accuracy']:.4%} "
                            f"s2_top5={validation['s2_top5_accuracy']:.4%} "
                            f"s1_entropy={validation['s1_mean_entropy']:.4f} "
                            f"s2_entropy={validation['s2_mean_entropy']:.4f}"
                        )
                        if "teacher_s2_loss" in validation:
                            print(
                                f"validation_conditioning step={global_step} "
                                f"predicted_s2={validation['s2_loss']:.6f} "
                                f"teacher_s2={validation['teacher_s2_loss']:.6f} "
                                f"mixed_s2={validation['mixed_s2_loss']:.6f} "
                                f"teacher_top1="
                                f"{validation['teacher_s2_top1_accuracy']:.4%}"
                            )
                    if validation["loss"] < best_validation_loss:
                        best_validation_loss = validation["loss"]
                        if distributed.is_main:
                            model_to_save = (
                                model.module if distributed.enabled else model
                            )
                            save_deployable_model(
                                model_to_save, best_dir, config, run_dir, fingerprint
                            )
                            artifact = (
                                "adapter"
                                if config.training.finetune_mode == "lora"
                                else "model"
                            )
                            print(f"Saved best {artifact} to {best_dir}")
                    if is_significant_validation_improvement(
                        validation["loss"],
                        early_stopping_reference_loss,
                        config.training.early_stopping_min_delta,
                    ):
                        early_stopping_reference_loss = validation["loss"]
                        stale_epochs = 0
                        validation_improved_this_epoch = True
                    # evaluate() switches the module to eval mode. Continue the
                    # remaining training batches with dropout and train behavior.
                    model.train()

                if global_step % config.training.checkpoint_interval == 0:
                    save_resume_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        run_dir,
                        distributed,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        early_stopping_reference_loss=(
                            early_stopping_reference_loss
                        ),
                        stale_epochs=stale_epochs,
                        validation_improved_this_epoch=(
                            validation_improved_this_epoch
                        ),
                        history_count=len(history),
                        metric_count=len(metric_records),
                        running_loss=running_loss,
                        running_s1_loss=running_s1_loss,
                        running_s2_loss=running_s2_loss,
                        running_batches=running_batches,
                        signature=signature,
                    )

        if running_batches:
            log_totals = torch.tensor(
                [
                    running_loss,
                    running_s1_loss,
                    running_s2_loss,
                    running_batches,
                ],
                dtype=torch.float64,
                device=device,
            )
            if distributed.enabled:
                dist.all_reduce(log_totals, op=dist.ReduceOp.SUM)
            if distributed.is_main:
                denominator = log_totals[3].item()
                train_loss = log_totals[0].item() / denominator
                train_s1_loss = log_totals[1].item() / denominator
                train_s2_loss = log_totals[2].item() / denominator
                metric_record = {
                    "split": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": train_loss,
                    "s1_loss": train_s1_loss,
                    "s2_loss": train_s2_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                metric_records.append(metric_record)
                append_jsonl(metrics_path, metric_record)
                write_loss_svg(loss_plot_path, metric_records)
                write_training_plots(run_dir / "plots", metric_records)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={train_loss:.6f} "
                    f"s1={train_s1_loss:.6f} "
                    f"s2={train_s2_loss:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                )
            running_loss = 0.0
            running_s1_loss = 0.0
            running_s2_loss = 0.0
            running_batches = 0

        validation = evaluate(
            model,
            tokenizer,
            loaders["validation"],
            device,
            split="validation",
            epoch=epoch,
            precision=precision,
            s1_loss_weight=config.training.s1_loss_weight,
            s2_loss_weight=config.training.s2_loss_weight,
            s2_conditioning=config.training.s2_conditioning,
            s2_teacher_loss_weight=config.training.s2_teacher_loss_weight,
            s2_predicted_loss_weight=(
                config.training.s2_predicted_loss_weight
            ),
            s2_top_k=config.training.s2_top_k,
            s2_temperature=config.training.s2_temperature,
            distributed=distributed,
        )
        if not math.isfinite(validation["loss"]):
            raise FloatingPointError(
                f"Non-finite validation loss at epoch={epoch}."
            )
        record = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            "validation_loss": validation["loss"],
            "validation_s1_loss": validation["s1_loss"],
            "validation_s2_loss": validation["s2_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"validation_{key}": validation[key]
                for key in PREDICTION_METRIC_KEYS
            },
            **{
                f"validation_{key}": validation[key]
                for key in MIXED_CONDITIONING_METRIC_KEYS
                if key in validation
            },
        }
        history.append(record)
        if distributed.is_main:
            validation_record: dict[str, float | int | str] = {
                "split": "validation",
                "validation_scope": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "loss": validation["loss"],
                "s1_loss": validation["s1_loss"],
                "s2_loss": validation["s2_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{key: validation[key] for key in PREDICTION_METRIC_KEYS},
                **{
                    key: validation[key]
                    for key in MIXED_CONDITIONING_METRIC_KEYS
                    if key in validation
                },
            }
            metric_records.append(validation_record)
            append_jsonl(metrics_path, validation_record)
            save_json(run_dir / "history.json", history)
            write_loss_svg(loss_plot_path, metric_records)
            write_training_plots(run_dir / "plots", metric_records)
            print(
                f"epoch={epoch} validation_loss={validation['loss']:.6f} "
                f"s1={validation['s1_loss']:.6f} s2={validation['s2_loss']:.6f}"
            )
            print(
                f"validation_tokens epoch={epoch} "
                f"s1_top1={validation['s1_top1_accuracy']:.4%} "
                f"s1_top5={validation['s1_top5_accuracy']:.4%} "
                f"s2_top1={validation['s2_top1_accuracy']:.4%} "
                f"s2_top5={validation['s2_top5_accuracy']:.4%} "
                f"s1_entropy={validation['s1_mean_entropy']:.4f} "
                f"s2_entropy={validation['s2_mean_entropy']:.4f}"
            )
            if "teacher_s2_loss" in validation:
                print(
                    f"validation_conditioning epoch={epoch} "
                    f"predicted_s2={validation['s2_loss']:.6f} "
                    f"teacher_s2={validation['teacher_s2_loss']:.6f} "
                    f"mixed_s2={validation['mixed_s2_loss']:.6f} "
                    f"teacher_top1="
                    f"{validation['teacher_s2_top1_accuracy']:.4%}"
                )

        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            if distributed.is_main:
                model_to_save = model.module if distributed.enabled else model
                save_deployable_model(
                    model_to_save, best_dir, config, run_dir, fingerprint
                )
                artifact = (
                    "adapter"
                    if config.training.finetune_mode == "lora"
                    else "model"
                )
                print(f"Saved best {artifact} to {best_dir}")
        if is_significant_validation_improvement(
            validation["loss"],
            early_stopping_reference_loss,
            config.training.early_stopping_min_delta,
        ):
            early_stopping_reference_loss = validation["loss"]
            validation_improved_this_epoch = True
        if validation_improved_this_epoch:
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                if distributed.is_main:
                    print("Early stopping triggered.")
                break

    if distributed.is_main:
        model_to_save = model.module if distributed.enabled else model
        save_deployable_model(
            model_to_save, last_dir, config, run_dir, fingerprint
        )
    if distributed.enabled:
        dist.barrier()
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if config.training.finetune_mode == "full":
        best_model = load_full_finetune_model(best_dir).to(device).eval()
    else:
        best_model = load_qv_lora_adapter(config.lora, best_dir).to(device).eval()
    test_metrics = evaluate(
        best_model,
        tokenizer,
        loaders["test"],
        device,
        split="test",
        precision=precision,
        s1_loss_weight=config.training.s1_loss_weight,
        s2_loss_weight=config.training.s2_loss_weight,
        s2_conditioning=config.training.s2_conditioning,
        s2_teacher_loss_weight=config.training.s2_teacher_loss_weight,
        s2_predicted_loss_weight=config.training.s2_predicted_loss_weight,
        s2_top_k=config.training.s2_top_k,
        s2_temperature=config.training.s2_temperature,
        distributed=distributed,
    )
    if distributed.is_main:
        conflict_summary = summarize_gradient_conflicts(metric_records)
        test_record: dict[str, float | int | str] = {
            "split": "test",
            "s2_conditioning": config.training.s2_conditioning,
            "epoch": len(history),
            "global_step": global_step,
            "loss": test_metrics["loss"],
            "s1_loss": test_metrics["s1_loss"],
            "s2_loss": test_metrics["s2_loss"],
            **{key: test_metrics[key] for key in PREDICTION_METRIC_KEYS},
            **{
                key: test_metrics[key]
                for key in MIXED_CONDITIONING_METRIC_KEYS
                if key in test_metrics
            },
        }
        metric_records.append(test_record)
        append_jsonl(metrics_path, test_record)
        save_json(
            run_dir / "result.json",
            {
                "best_validation_loss": best_validation_loss,
                "test_loss": test_metrics["loss"],
                "test_s1_loss": test_metrics["s1_loss"],
                "test_s2_loss": test_metrics["s2_loss"],
                **{
                    f"test_{key}": test_metrics[key]
                    for key in PREDICTION_METRIC_KEYS
                },
                **{
                    f"test_{key}": test_metrics[key]
                    for key in MIXED_CONDITIONING_METRIC_KEYS
                    if key in test_metrics
                },
                "test_targets": test_metrics["targets"],
                "epochs_completed": len(history),
                "global_steps": global_step,
                "learning_rate": config.training.learning_rate,
                "warmup_steps": config.training.warmup_steps,
                "resolved_precision": precision,
                "world_size": distributed.world_size,
                "finetune_mode": config.training.finetune_mode,
                "s2_conditioning": config.training.s2_conditioning,
                "s2_teacher_loss_weight": (
                    config.training.s2_teacher_loss_weight
                ),
                "s2_predicted_loss_weight": (
                    config.training.s2_predicted_loss_weight
                ),
                "s2_top_k": config.training.s2_top_k,
                "s2_temperature": config.training.s2_temperature,
                "target_mode": config.data.target_mode,
                "target_length": config.data.target_length,
                "initialized_from_adapter": (
                    str(initialization_source)
                    if initialization_source is not None
                    else None
                ),
                "lora_rank": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "proj_s1_rank": config.lora.proj_s1_rank,
                "proj_s1_alpha": config.lora.proj_s1_alpha,
                "full_finetune_proj_s1": config.lora.full_finetune_proj_s1,
                "dependency_rank": config.lora.dependency_rank,
                "dependency_alpha": config.lora.dependency_alpha,
                "fusion_rank": config.lora.fusion_rank,
                "fusion_alpha": config.lora.fusion_alpha,
                "ffn_rank": config.lora.ffn_rank,
                "ffn_alpha": config.lora.ffn_alpha,
                "target_modules": list(config.lora.target_modules),
                "exclude_modules": list(config.lora.exclude_modules),
                "base_model_path": str(config.lora.base_model_path.resolve()),
                "tokenizer_path": str(config.lora.tokenizer_path.resolve()),
                "full_trainable_modules": (
                    ["head.proj_s1"]
                    if config.lora.full_finetune_proj_s1
                    else []
                ),
                "run_directory": str(run_dir),
                "last_model": str(last_dir),
                "best_model": str(best_dir),
                # Backward-compatible result keys retained for LoRA consumers.
                "last_adapter": (
                    str(last_dir) if config.training.finetune_mode == "lora" else None
                ),
                "best_adapter": (
                    str(best_dir) if config.training.finetune_mode == "lora" else None
                ),
                "gradient_conflict_summary": conflict_summary,
            },
        )
        if conflict_summary is not None:
            print(
                "gradient_conflict_summary "
                f"samples={conflict_summary['sample_count']} "
                f"median_cos={conflict_summary['median_cosine']:.4f} "
                f"negative_ratio="
                f"{conflict_summary['negative_cosine_ratio']:.3f} "
                f"strong_conflict_ratio="
                f"{conflict_summary['strong_conflict_ratio']:.3f} "
                f"median_s2_to_s1="
                f"{conflict_summary['median_s2_to_s1_ratio']:.3f}"
            )
    return history
