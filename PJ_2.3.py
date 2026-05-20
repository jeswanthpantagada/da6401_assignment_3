import os
import csv
import math
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    "output_dir": "attention_outputs",
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
# LOSS
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


def build_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: tok for tok, idx in vocab.items()}


def ids_to_sentence(ids: List[int], inv_vocab: Dict[int, str]) -> str:
    tokens = []
    for idx in ids:
        if idx in (PAD_IDX, SOS_IDX, EOS_IDX):
            continue
        tokens.append(inv_vocab.get(idx, "<unk>"))
    return " ".join(tokens)


# ============================================================
# HELPERS
# ============================================================

def save_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int = SOS_IDX,
    end_symbol: int = EOS_IDX,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = src.device

    model.eval()
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if torch.all(next_token.squeeze(1) == end_symbol):
            break

    return ys


def get_grad_norm(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    return tensor.detach().norm(p=2).item()


def entropy_from_attention(attn: torch.Tensor) -> float:
    """
    attn: [L, L]
    """
    eps = 1e-9
    p = attn.clamp_min(eps)
    return float((-p * p.log()).sum(dim=-1).mean().item())


def diagonal_mass(attn: torch.Tensor) -> float:
    """
    Mean attention mass on the diagonal (same-position attention).
    attn: [L, L]
    """
    if attn.size(0) != attn.size(1):
        return 0.0
    return float(attn.diagonal().mean().item())


def mean_attention_distance(attn: torch.Tensor) -> float:
    """
    Average absolute query-key distance attended to.
    attn: [L, L]
    """
    L = attn.size(0)
    pos = torch.arange(L, device=attn.device)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
    return float((attn * dist).sum().item() / max(L, 1))


# ============================================================
# TRAINING / VALIDATION
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


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_de: Dict[str, int],
    vocab_en: Dict[str, int],
    device: torch.device,
) -> Tuple[Transformer, List[Dict]]:
    """
    Trains the baseline transformer for 15 epochs and returns epoch metrics.
    """
    # Prevent automatic checkpoint loading from model.py
    if hasattr(Transformer, "_try_autoload_checkpoint"):
        Transformer._try_autoload_checkpoint = lambda self: None

    model = Transformer(
        src_vocab_size=len(vocab_de),
        tgt_vocab_size=len(vocab_en),
        d_model=CONFIG["d_model"],
        N=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    ).to(device)

    model.src_vocab = vocab_de
    model.tgt_vocab = vocab_en
    model.inv_tgt_vocab = build_inverse_vocab(vocab_en)

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

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
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
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_acc:.4f}"
        )

    return model, epoch_rows


# ============================================================
# ATTENTION EXTRACTION
# ============================================================

@torch.no_grad()
def get_sample_and_attention(
    model: Transformer,
    val_loader: DataLoader,
    vocab_de: Dict[str, int],
    vocab_en: Dict[str, int],
    device: torch.device,
):
    """
    Extracts attention from the last encoder layer for one validation sample.
    Returns:
        src_tokens, tgt_tokens, pred_tokens, attention_tensor
    """
    inv_de = build_inverse_vocab(vocab_de)
    inv_en = build_inverse_vocab(vocab_en)

    sample_src, sample_tgt = next(iter(val_loader))
    sample_src = sample_src.to(device)
    sample_tgt = sample_tgt.to(device)

    single_src = sample_src[0:1]
    single_tgt = sample_tgt[0:1]

    src_mask = make_src_mask(single_src, pad_idx=PAD_IDX).to(device)

    model.eval()
    _ = model.encode(single_src, src_mask)

    attn = model.encoder.layers[-1].self_attn.last_attn_weights
    if attn is None:
        raise RuntimeError("Attention weights were not captured.")

    # attn shape: [batch, heads, seq_len, seq_len]
    attn = attn[0].detach().cpu()

    pred_ids = greedy_decode(
        model=model,
        src=single_src,
        src_mask=src_mask,
        max_len=CONFIG["max_decode_len"],
        start_symbol=SOS_IDX,
        end_symbol=EOS_IDX,
        device=device,
    )[0].tolist()

    src_tokens = [inv_de.get(i, "<unk>") for i in single_src[0].tolist()]
    tgt_tokens = [inv_en.get(i, "<unk>") for i in single_tgt[0].tolist()]
    pred_tokens = [inv_en.get(i, "<unk>") for i in pred_ids]

    return src_tokens, tgt_tokens, pred_tokens, attn


# ============================================================
# VISUALIZATION
# ============================================================

def save_attention_csv(attn: torch.Tensor, tokens: List[str], out_csv: str) -> None:
    """
    Saves attention in long-form CSV:
    head, query_token, key_token, query_index, key_index, weight
    """
    rows = []
    num_heads, seq_len, _ = attn.shape

    for head in range(num_heads):
        for q in range(seq_len):
            for k in range(seq_len):
                rows.append({
                    "head": head,
                    "query_index": q,
                    "key_index": k,
                    "query_token": tokens[q],
                    "key_token": tokens[k],
                    "weight": float(attn[head, q, k].item()),
                })

    save_csv(
        out_csv,
        rows,
        fieldnames=["head", "query_index", "key_index", "query_token", "key_token", "weight"],
    )


def save_attention_summary_csv(attn: torch.Tensor, out_csv: str) -> None:
    """
    Saves per-head summary statistics useful for the report.
    """
    rows = []
    num_heads = attn.shape[0]

    for head in range(num_heads):
        A = attn[head]
        rows.append({
            "head": head,
            "mean_entropy": entropy_from_attention(A),
            "mean_diagonal_mass": diagonal_mass(A),
            "mean_attention_distance": mean_attention_distance(A),
            "max_attention_weight": float(A.max().item()),
        })

    save_csv(
        out_csv,
        rows,
        fieldnames=[
            "head",
            "mean_entropy",
            "mean_diagonal_mass",
            "mean_attention_distance",
            "max_attention_weight",
        ],
    )


def plot_attention_heads(attn: torch.Tensor, tokens: List[str], output_dir: str) -> None:
    """
    Saves one heatmap per head and a combined grid figure.
    """
    num_heads = attn.shape[0]

    # Combined grid figure
    cols = 2
    rows = math.ceil(num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten() if num_heads > 1 else [axes]

    for head in range(num_heads):
        ax = axes[head]
        im = ax.imshow(attn[head].numpy(), aspect="auto", interpolation="nearest")
        ax.set_title(f"Head {head}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(num_heads, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_heads_grid.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Individual head figures
    for head in range(num_heads):
        plt.figure(figsize=(10, 8))
        plt.imshow(attn[head].numpy(), aspect="auto", interpolation="nearest")
        plt.title(f"Last Encoder Layer Attention - Head {head}")
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
        plt.yticks(range(len(tokens)), tokens, fontsize=8)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention_head_{head}.png"), dpi=200, bbox_inches="tight")
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, vocab_de, vocab_en = build_loaders()

    # Train model for 15 epochs
    model, epoch_rows = train_model(train_loader, val_loader, vocab_de, vocab_en, device)

    # Save training CSV
    save_csv(
        os.path.join(CONFIG["output_dir"], "epoch_metrics.csv"),
        epoch_rows,
        fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"],
    )

    # Plot training curves
    epochs = [r["epoch"] for r in epoch_rows]
    train_loss = [r["train_loss"] for r in epoch_rows]
    val_loss = [r["val_loss"] for r in epoch_rows]
    val_acc = [r["val_accuracy"] for r in epoch_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "val_accuracy_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Extract one sample attention
    src_tokens, tgt_tokens, pred_tokens, attn = get_sample_and_attention(
        model=model,
        val_loader=val_loader,
        vocab_de=vocab_de,
        vocab_en=vocab_en,
        device=device,
    )

    # Save sample translation CSV
    sample_rows = [{
        "source_sentence": " ".join(src_tokens),
        "reference_translation": " ".join(tgt_tokens),
        "predicted_translation": " ".join(pred_tokens),
    }]
    save_csv(
        os.path.join(CONFIG["output_dir"], "sample_translation.csv"),
        sample_rows,
        fieldnames=["source_sentence", "reference_translation", "predicted_translation"],
    )

    # Save attention CSVs
    save_attention_csv(
        attn=attn,
        tokens=src_tokens,
        out_csv=os.path.join(CONFIG["output_dir"], "last_encoder_attention_long.csv"),
    )
    save_attention_summary_csv(
        attn=attn,
        out_csv=os.path.join(CONFIG["output_dir"], "last_encoder_attention_summary.csv"),
    )

    # Save attention heatmaps
    plot_attention_heads(attn=attn, tokens=src_tokens, output_dir=CONFIG["output_dir"])

    print("\nDone.")
    print(f"All CSV and PNG files are saved in: {CONFIG['output_dir']}")
    print("Important files created:")
    print(f"- {os.path.join(CONFIG['output_dir'], 'epoch_metrics.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'sample_translation.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'last_encoder_attention_long.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'last_encoder_attention_summary.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'attention_heads_grid.png')}")


if __name__ == "__main__":
    main()