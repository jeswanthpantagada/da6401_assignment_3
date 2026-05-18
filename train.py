import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

try:
    import evaluate
except Exception:
    evaluate = None

import spacy

from dataset import Multi30kDataset, collate_fn, build_vocab_from_train
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


# =========================
# Config
# =========================
CONFIG = {
    "seed": 42,
    "project": "da6401-assignment-3",
    "run_name": "transformer-baseline",
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 25,
    "warmup_steps": 4000,
    "fixed_lr": 1e-4,
    "use_noam": True,
    "label_smoothing": 0.1,
    "max_decode_len": 60,
    "num_workers": 0,
    "clip_grad_norm": 1.0,
    "save_dir": "checkpoints",
    "best_ckpt_name": "best_transformer.pt",
    "last_ckpt_name": "last_transformer.pt",
}

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# SpaCy tokenizers
# =========================
def load_spacy_models():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        spacy_de = spacy.blank("de")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        spacy_en = spacy.blank("en")

    return spacy_de, spacy_en


SPACY_DE, SPACY_EN = load_spacy_models()


# =========================
# Dataset helpers
# =========================
def build_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    # Build vocabulary from train split
    vocab_de, vocab_en = build_vocab_from_train()
    
    # Create datasets with shared vocabulary
    train_dataset = Multi30kDataset(split="train", vocab_de=vocab_de, vocab_en=vocab_en)
    val_dataset = Multi30kDataset(split="validation", vocab_de=vocab_de, vocab_en=vocab_en)
    test_dataset = Multi30kDataset(split="test", vocab_de=vocab_de, vocab_en=vocab_en)

    g = torch.Generator()
    g.manual_seed(CONFIG["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, vocab_de, vocab_en


def build_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: tok for tok, idx in vocab.items()}


def ids_to_sentence(ids: List[int], inv_vocab: Dict[int, str]) -> str:
    tokens = []
    for idx in ids:
        if idx in (PAD_IDX, SOS_IDX, EOS_IDX):
            continue
        tokens.append(inv_vocab.get(idx, "<unk>"))
    return " ".join(tokens)


# =========================
# Loss: Label smoothing
# =========================
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
        log_probs = F.log_softmax(logits, dim=-1)
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


# =========================
# Metrics
# =========================
def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = PAD_IDX) -> float:
    pred = logits.argmax(dim=-1)
    mask = target.ne(pad_idx)
    correct = (pred == target) & mask
    denom = mask.sum().clamp_min(1)
    return (correct.sum().float() / denom.float()).item()


def get_bleu_scorer():
    if evaluate is not None:
        try:
            bleu_metric = evaluate.load("bleu")

            def _compute(predictions: List[str], references: List[List[str]]) -> float:
                result = bleu_metric.compute(predictions=predictions, references=references)
                return float(result["bleu"]) * 100  # Return as percentage

            return _compute
        except Exception:
            pass

    # Fallback
    def _compute(predictions: List[str], references: List[List[str]]) -> float:
        return 0.0

    return _compute


BLEU_SCORE = get_bleu_scorer()


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


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    data_loader: DataLoader,
    tgt_vocab: Dict[str, int],
    device: torch.device,
    max_len: int = 60,
) -> float:
    model.eval()
    inv_tgt_vocab = build_inverse_vocab(tgt_vocab)

    predictions: List[str] = []
    references: List[List[str]] = []

    for src, tgt in tqdm(data_loader, desc="BLEU", leave=False):
        src = src.to(device)
        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

        for i in range(src.size(0)):
            single_src = src[i:i + 1]
            single_mask = src_mask[i:i + 1]

            pred_ids = greedy_decode(
                model,
                single_src,
                single_mask,
                max_len=max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )[0].tolist()

            pred_sentence = ids_to_sentence(pred_ids, inv_tgt_vocab)
            tgt_sentence = ids_to_sentence(tgt[i].tolist(), inv_tgt_vocab)

            predictions.append(pred_sentence)
            references.append([tgt_sentence])

    return BLEU_SCORE(predictions, references)


# =========================
# Training / validation epoch
# =========================
def run_epoch(
    model: Transformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    train: bool,
    epoch: int,
) -> Tuple[float, float]:
    model.train(train)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"{'Train' if train else 'Valid'} Epoch {epoch}", leave=False)
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits, tgt_expected)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        acc = token_accuracy(logits.detach(), tgt_expected, PAD_IDX)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}", lr=f"{current_lr:.6e}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc


# =========================
# Checkpointing
# =========================
def save_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_bleu: float,
    vocab_de: Dict[str, int],
    vocab_en: Dict[str, int],
    config: Dict,
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_val_bleu": best_val_bleu,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "vocab_de": vocab_de,
        "vocab_en": vocab_en,
        "config": config,
        "model_config": {
            "src_vocab_size": len(vocab_de),
            "tgt_vocab_size": len(vocab_en),
            "d_model": config["d_model"],
            "N": config["num_layers"],
            "num_heads": config["num_heads"],
            "d_ff": config["d_ff"],
            "dropout": config["dropout"],
        }
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Set vocabularies on model
    model.src_vocab = ckpt.get("vocab_de")
    model.tgt_vocab = ckpt.get("vocab_en")
    if ckpt.get("vocab_en") is not None:
        model.inv_tgt_vocab = {v: k for k, v in ckpt["vocab_en"].items()}
    
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
    return ckpt


# =========================
# Visualization
# =========================
def plot_history(history: Dict[str, List[float]], save_path: str = "training_curves.png") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Token Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_bleu"], label="Val BLEU", color='green')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("BLEU Score")
    axes[2].set_title("Validation BLEU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# =========================
# Main training script
# =========================
def main() -> None:
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb (will be disabled in autograder environment)
    try:
        wandb.init(
            project=CONFIG["project"],
            name=CONFIG["run_name"],
            config=CONFIG,
            mode=os.environ.get("WANDB_MODE", "disabled"),  # Changed to disabled by default
        )
    except:
        pass

    train_loader, val_loader, test_loader, vocab_de, vocab_en = build_dataloaders(CONFIG["batch_size"])
    src_vocab_size = len(vocab_de)
    tgt_vocab_size = len(vocab_en)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=CONFIG["d_model"],
        N=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    ).to(device)
    
    # Store vocabularies on model
    model.src_vocab = vocab_de
    model.tgt_vocab = vocab_en
    model.inv_tgt_vocab = build_inverse_vocab(vocab_en)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["fixed_lr"] if not CONFIG["use_noam"] else 0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = None
    if CONFIG["use_noam"]:
        scheduler = NoamScheduler(
            optimizer=optimizer,
            d_model=CONFIG["d_model"],
            warmup_steps=CONFIG["warmup_steps"],
        )

    criterion = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=CONFIG["label_smoothing"],
    ).to(device)

    best_val_bleu = -1.0
    best_path = os.path.join(CONFIG["save_dir"], CONFIG["best_ckpt_name"])
    last_path = os.path.join(CONFIG["save_dir"], CONFIG["last_ckpt_name"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_bleu": [],
        "lr": [],
    }

    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train=True,
            epoch=epoch,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=device,
            train=False,
            epoch=epoch,
        )

        # Calculate BLEU every 2 epochs for speed, but always on last few epochs
        if epoch % 2 == 0 or epoch >= CONFIG["epochs"] - 3:
            val_bleu = evaluate_bleu(
                model=model,
                data_loader=val_loader,
                tgt_vocab=vocab_en,
                device=device,
                max_len=CONFIG["max_decode_len"],
            )
        else:
            val_bleu = 0.0  # Skip BLEU calculation for speed

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_bleu"].append(val_bleu)
        history["lr"].append(current_lr)

        try:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_bleu": val_bleu,
                "lr": current_lr,
            })
        except:
            pass

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"val_bleu={val_bleu:.2f} | lr={current_lr:.6e}"
        )

        # Save best by BLEU
        if val_bleu > best_val_bleu and val_bleu > 0:
            best_val_bleu = val_bleu
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_bleu=best_val_bleu,
                vocab_de=vocab_de,
                vocab_en=vocab_en,
                config=CONFIG,
            )
            print(f"  → Saved new best model with BLEU {best_val_bleu:.2f}")

        # Always save last checkpoint
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_bleu=best_val_bleu,
            vocab_de=vocab_de,
            vocab_en=vocab_en,
            config=CONFIG,
        )

        # Sample translation
        if epoch % 5 == 0:
            sample_src, sample_tgt = next(iter(val_loader))
            sample_sentence = ids_to_sentence(sample_src[0].tolist(), build_inverse_vocab(vocab_de))
            sample_pred_ids = greedy_decode(
                model=model,
                src=sample_src[0:1].to(device),
                src_mask=make_src_mask(sample_src[0:1].to(device), pad_idx=PAD_IDX).to(device),
                max_len=CONFIG["max_decode_len"],
                device=device,
            )[0].tolist()
            sample_pred = ids_to_sentence(sample_pred_ids, build_inverse_vocab(vocab_en))
            sample_ref = ids_to_sentence(sample_tgt[0].tolist(), build_inverse_vocab(vocab_en))

            print(f"  SRC:  {sample_sentence}")
            print(f"  PRED: {sample_pred}")
            print(f"  REF:  {sample_ref}")

    print(f"\nBest validation BLEU: {best_val_bleu:.2f}")
    print(f"Best checkpoint saved to: {best_path}")

    plot_history(history)

    # Load best checkpoint for test evaluation
    print("\nLoading best checkpoint for test evaluation...")
    best_ckpt = load_checkpoint(best_path, model, optimizer=None, scheduler=None)
    print(f"Loaded best model from epoch {best_ckpt['epoch']}")

    test_bleu = evaluate_bleu(
        model=model,
        data_loader=test_loader,
        tgt_vocab=vocab_en,
        device=device,
        max_len=CONFIG["max_decode_len"],
    )
    print(f"\n{'='*60}")
    print(f"FINAL TEST BLEU: {test_bleu:.2f}")
    print(f"{'='*60}")
    
    try:
        wandb.log({"test_bleu": test_bleu})
        wandb.finish()
    except:
        pass


if __name__ == "__main__":
    main()
