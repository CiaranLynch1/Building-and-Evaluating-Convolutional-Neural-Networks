import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from cifar_data import make_loaders, prepare_inputs, set_seed

CHECKPOINT_DIR = Path("checkpoints")  # already in .gitignore


@dataclass
class Run:
    # same idea as the Run dataclass in Part A5
    name: str
    options: dict = field(
        default_factory=dict
    )  # passed to the model, e.g. {"width": 32}
    seed: int = 42
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    l2: float = 5e-4  # weight decay = L2 regularisation strength (Task 3)
    augmentation: bool = False  # off for Task 2/3 baselines, on in Task 4


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate(model, loader, stats, device, size=32):
    model.eval()  # turns off dropout, BatchNorm uses running stats
    loss_sum, correct, count = 0.0, 0, 0
    for raw, targets, _ in loader:
        raw, targets = raw.to(device), targets.to(device)
        logits = model(prepare_inputs(raw, stats, size))
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(targets)
    return {"ce": loss_sum / count, "accuracy": correct / count}


def train_one_epoch(model, loader, optimizer, stats, device, size=32):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for raw, targets, _ in loader:
        raw, targets = raw.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prepare_inputs(raw, stats, size))
        loss = nn.functional.cross_entropy(logits, targets)
        if not math.isfinite(loss.item()):
            raise RuntimeError(
                "loss became NaN/inf - learning rate is probably too high"
            )
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(targets)
    return {"ce": loss_sum / count, "accuracy": correct / count}


def train(run, model, loaders, stats, device, size=32, save=True):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=run.lr, momentum=run.momentum, weight_decay=run.l2
    )
    # cosine decays the lr to ~0 over exactly run.epochs (StepLR(30) never fired with 10 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = {"val_accuracy": -1.0, "epoch": None, "state": None}
    started = time.perf_counter()

    for epoch in range(1, run.epochs + 1):
        epoch_start = time.perf_counter()
        training = train_one_epoch(
            model, loaders["train"], optimizer, stats, device, size
        )
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        validation = evaluate(model, loaders["val"], stats, device, size)

        row = {
            "epoch": epoch,
            "lr": lr_used,
            "train_ce": training["ce"],
            "train_accuracy": training["accuracy"],
            "val_ce": validation["ce"],
            "val_accuracy": validation["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"[{epoch}/{run.epochs}] {run.name} | train ce {row['train_ce']:.3f} "
            f"acc {row['train_accuracy']:.3f} | val ce {row['val_ce']:.3f} "
            f"acc {row['val_accuracy']:.3f} | {row['epoch_seconds']:.1f}s"
        )

        if validation["accuracy"] > best["val_accuracy"]:
            best = {
                "val_accuracy": validation["accuracy"],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }

    model.load_state_dict(
        best["state"]
    )  # return the selected checkpoint, not the last epoch

    summary = {
        "run": run.name,
        "parameters": count_parameters(model),
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "train_seconds": time.perf_counter() - started,
        "samples_seen": run.epochs * len(loaders["train"].dataset),
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "config": asdict(run),
    }

    if save:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / f"{run.name}_seed{run.seed}.pt"
        torch.save(
            {"state_dict": best["state"], "summary": summary, "history": history}, path
        )
        summary["checkpoint"] = str(path)

    return history, summary


def run_experiment(run, model_class, data, split, device, num_workers=0):
    # seed -> build model -> loaders -> train, always in that order so runs are reproducible
    set_seed(run.seed)
    model = model_class(**run.options).to(device)
    loaders = make_loaders(
        run,
        data["train"],
        split["train_ids"],
        split["val_ids"],
        num_workers=num_workers,
    )
    history, summary = train(run, model, loaders, split["stats"], device)
    return model, history, summary
