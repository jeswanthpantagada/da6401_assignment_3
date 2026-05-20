import os
import csv
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as dataset_module
import model as model_module
from dataset import Multi30kDataset, collate_fn, build_vocab_from_train
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "seed": 42,
    "epochs": 15,
    "batch_size": 64,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "clip_grad_norm": 1.0,
    "max_decode_len": 60,
    "num_workers": 0,
    "output_dir": "scaling_factor_outputs",
}

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# LABEL SMOOTHING LOSS
# ============================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = PAD_IDX, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch, seq_len, vocab]
        target: [batch, seq_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        non_pad = target.ne(self.pad_idx)
        n_valid = non_pad.sum().clamp_min(1)

        true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        target_safe = target.clone()
        target_safe[~non_pad] = self.pad_idx

        true_dist.scatter_(1, target_safe.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0.0
        true_dist[~non_pad] = 0.0

        loss = self.criterion(log_probs, true_dist)
        return loss / n_valid


# ============================================================
# DATA
# ============================================================

def build_loaders() -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    vocab_de, vocab_en = build_vocab_from_train()

    train_dataset = Multi30kDataset(split="train", vocab_de=vocab_de, vocab_en=vocab_en)
    val_dataset = Multi30kDataset(split="validation", vocab_de=vocab_de, vocab_en=vocab_en)

    g = torch.Generator()
    g.manual_seed(CONFIG["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, vocab_de, vocab_en


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def evaluate_accuracy(model: Transformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        pred = output.argmax(dim=-1)
        mask = tgt_expected != PAD_IDX

        total_correct += ((pred == tgt_expected) & mask).sum().item()
        total_tokens += mask.sum().item()

    return total_correct / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model: Transformer, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output, tgt_expected)

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def get_grad_norm(tensor: torch.Tensor) -> float:
    if tensor is None:
        return 0.0
    return tensor.detach().norm(p=2).item()


# ============================================================
# CSV HELPERS
# ============================================================

def save_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# ATTENTION PATCHING
# ============================================================

def scaled_attention_with_scaling(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights_for_output = dropout(attn_weights) if dropout is not None else attn_weights
    output = torch.matmul(attn_weights_for_output, V)
    return output, attn_weights


def scaled_attention_without_scaling(Q, K, V, mask=None, dropout=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights_for_output = dropout(attn_weights) if dropout is not None else attn_weights
    output = torch.matmul(attn_weights_for_output, V)
    return output, attn_weights


# ============================================================
# TRAIN ONE RUN
# ============================================================

def train_one_run(
    run_name: str,
    use_scaling: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_de: Dict[str, int],
    vocab_en: Dict[str, int],
    device: torch.device,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        epoch_rows: per-epoch metrics
        grad_rows: per-step gradient norms for first 1000 steps
    """

    # Disable automatic checkpoint loading from model.py so every run starts fresh
    if hasattr(Transformer, "_try_autoload_checkpoint"):
        Transformer._try_autoload_checkpoint = lambda self: None

    # Patch the attention function globally inside model.py
    if use_scaling:
        model_module.scaled_dot_product_attention = scaled_attention_with_scaling
    else:
        model_module.scaled_dot_product_attention = scaled_attention_without_scaling

    model = Transformer(
        src_vocab_size=len(vocab_de),
        tgt_vocab_size=len(vocab_en),
        d_model=CONFIG["d_model"],
        N=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    ).to(device)

    # Make sure model starts fresh even if anything was loaded unexpectedly
    if hasattr(model, "_reset_parameters"):
        model._reset_parameters()

    model.src_vocab = vocab_de
    model.tgt_vocab = vocab_en
    model.inv_tgt_vocab = {v: k for k, v in vocab_en.items()}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=CONFIG["d_model"],
        warmup_steps=CONFIG["warmup_steps"],
    )

    criterion = LabelSmoothingLoss(
        vocab_size=len(vocab_en),
        pad_idx=PAD_IDX,
        smoothing=CONFIG["label_smoothing"],
    ).to(device)

    epoch_rows = []
    grad_rows = []
    global_step = 0

    print(f"\n==================================================")
    print(f"Running: {run_name}")
    print(f"Scaling factor: {'ON' if use_scaling else 'OFF'}")
    print(f"==================================================")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"{run_name} | Epoch {epoch}", leave=False)

        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output, tgt_expected)
            loss.backward()

            # Gradient norms for first 1000 steps only
            global_step += 1
            if global_step <= 1000:
                q_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
                k_grad = model.encoder.layers[0].self_attn.W_k.weight.grad

                grad_rows.append({
                    "run_name": run_name,
                    "step": global_step,
                    "grad_norm_q": get_grad_norm(q_grad),
                    "grad_norm_k": get_grad_norm(k_grad),
                })

            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_batches += 1

            pbar.set_postfix(
                train_loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6e}",
            )

        avg_train_loss = total_train_loss / max(train_batches, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        epoch_rows.append({
            "run_name": run_name,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"{run_name} | Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_acc:.4f}"
        )

    return epoch_rows, grad_rows


# ============================================================
# PLOTS
# ============================================================

def plot_comparison(epoch_noam: List[Dict], epoch_noscale: List[Dict]) -> None:
    epochs = [row["epoch"] for row in epoch_noam]

    noam_train = [row["train_loss"] for row in epoch_noam]
    noscale_train = [row["train_loss"] for row in epoch_noscale]

    noam_val_acc = [row["val_accuracy"] for row in epoch_noam]
    noscale_val_acc = [row["val_accuracy"] for row in epoch_noscale]

    noam_val_loss = [row["val_loss"] for row in epoch_noam]
    noscale_val_loss = [row["val_loss"] for row in epoch_noscale]

    # Training loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, noam_train, label="With scaling 1/sqrt(dk)")
    plt.plot(epochs, noscale_train, label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss: Scaling vs No Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_loss_comparison.png"), dpi=200)
    plt.close()

    # Validation accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, noam_val_acc, label="With scaling 1/sqrt(dk)")
    plt.plot(epochs, noscale_val_acc, label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy: Scaling vs No Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "validation_accuracy_comparison.png"), dpi=200)
    plt.close()

    # Validation loss comparison
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, noam_val_loss, label="With scaling 1/sqrt(dk)")
    plt.plot(epochs, noscale_val_loss, label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss: Scaling vs No Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "validation_loss_comparison.png"), dpi=200)
    plt.close()


def plot_grad_norms(grad_rows: List[Dict]) -> None:
    run_names = sorted(set(row["run_name"] for row in grad_rows))

    for grad_key, filename, title in [
        ("grad_norm_q", "grad_norm_q_comparison.png", "Query Weight Gradient Norm"),
        ("grad_norm_k", "grad_norm_k_comparison.png", "Key Weight Gradient Norm"),
    ]:
        plt.figure(figsize=(10, 5))
        for run_name in run_names:
            rows = [r for r in grad_rows if r["run_name"] == run_name]
            rows = sorted(rows, key=lambda x: x["step"])
            steps = [r["step"] for r in rows]
            vals = [r[grad_key] for r in rows]
            plt.plot(steps, vals, label=run_name)

        plt.xlabel("Step")
        plt.ylabel("Gradient Norm")
        plt.title(f"{title} (First 1000 Steps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], filename), dpi=200)
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, vocab_de, vocab_en = build_loaders()

    # Run 1: With scaling
    epoch_with_scaling, grad_with_scaling = train_one_run(
        run_name="with_scaling",
        use_scaling=True,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_de=vocab_de,
        vocab_en=vocab_en,
        device=device,
    )

    # Run 2: Without scaling
    epoch_without_scaling, grad_without_scaling = train_one_run(
        run_name="without_scaling",
        use_scaling=False,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_de=vocab_de,
        vocab_en=vocab_en,
        device=device,
    )

    # Save per-epoch CSVs
    save_csv(
        os.path.join(CONFIG["output_dir"], "with_scaling_epoch_metrics.csv"),
        epoch_with_scaling,
        fieldnames=["run_name", "epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"],
    )
    save_csv(
        os.path.join(CONFIG["output_dir"], "without_scaling_epoch_metrics.csv"),
        epoch_without_scaling,
        fieldnames=["run_name", "epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"],
    )

    # Save combined CSV
    combined_epoch_rows = []
    for row in epoch_with_scaling:
        combined_epoch_rows.append({"setting": "with_scaling", **row})
    for row in epoch_without_scaling:
        combined_epoch_rows.append({"setting": "without_scaling", **row})

    save_csv(
        os.path.join(CONFIG["output_dir"], "combined_epoch_metrics.csv"),
        combined_epoch_rows,
        fieldnames=["setting", "run_name", "epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"],
    )

    # Save gradient norm CSV
    combined_grad_rows = grad_with_scaling + grad_without_scaling
    save_csv(
        os.path.join(CONFIG["output_dir"], "gradient_norms_first_1000_steps.csv"),
        combined_grad_rows,
        fieldnames=["run_name", "step", "grad_norm_q", "grad_norm_k"],
    )

    # Overlay CSV for easier reporting
    overlay_rows = []
    for i in range(CONFIG["epochs"]):
        overlay_rows.append({
            "epoch": i + 1,
            "with_scaling_train_loss": epoch_with_scaling[i]["train_loss"],
            "without_scaling_train_loss": epoch_without_scaling[i]["train_loss"],
            "with_scaling_val_accuracy": epoch_with_scaling[i]["val_accuracy"],
            "without_scaling_val_accuracy": epoch_without_scaling[i]["val_accuracy"],
            "with_scaling_val_loss": epoch_with_scaling[i]["val_loss"],
            "without_scaling_val_loss": epoch_without_scaling[i]["val_loss"],
        })

    save_csv(
        os.path.join(CONFIG["output_dir"], "overlay_comparison.csv"),
        overlay_rows,
        fieldnames=[
            "epoch",
            "with_scaling_train_loss",
            "without_scaling_train_loss",
            "with_scaling_val_accuracy",
            "without_scaling_val_accuracy",
            "with_scaling_val_loss",
            "without_scaling_val_loss",
        ],
    )

    # Plots
    plot_comparison(epoch_with_scaling, epoch_without_scaling)
    plot_grad_norms(combined_grad_rows)

    print("\nDone.")
    print(f"All CSV and PNG files are saved in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()