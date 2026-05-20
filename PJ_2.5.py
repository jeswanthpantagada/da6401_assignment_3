import os
import csv
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import evaluate
except Exception:
    evaluate = None

import model as model_module
from dataset import Multi30kDataset, collate_fn, build_vocab_from_train
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
    "clip_grad_norm": 1.0,
    "max_decode_len": 60,
    "num_workers": 0,
    "output_dir": "label_smoothing_outputs",
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
    model,
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
        tgt_mask = model_module.make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if torch.all(next_token.squeeze(1) == end_symbol):
            break

    return ys


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = model_module.make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = model_module.make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        pred = output.argmax(dim=-1)
        mask = tgt_expected != PAD_IDX

        total_correct += ((pred == tgt_expected) & mask).sum().item()
        total_tokens += mask.sum().item()

    return total_correct / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = model_module.make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = model_module.make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output, tgt_expected)

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_prediction_confidence(model, loader: DataLoader, device: torch.device) -> float:
    """
    Mean probability assigned to the correct token on the validation set.
    """
    model.eval()
    total_conf = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_mask = model_module.make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = model_module.make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        probs = torch.softmax(logits, dim=-1)

        correct_probs = probs.gather(-1, tgt_expected.unsqueeze(-1)).squeeze(-1)
        mask = tgt_expected != PAD_IDX

        total_conf += correct_probs[mask].sum().item()
        total_tokens += mask.sum().item()

    return total_conf / max(total_tokens, 1)


def compute_bleu(model, loader: DataLoader, inv_tgt_vocab: Dict[int, str], device: torch.device) -> float:
    model.eval()

    predictions: List[str] = []
    references: List[List[str]] = []

    with torch.no_grad():
        for src, tgt in tqdm(loader, desc="BLEU", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = model_module.make_src_mask(src, pad_idx=PAD_IDX).to(device)

            for i in range(src.size(0)):
                single_src = src[i : i + 1]
                single_mask = src_mask[i : i + 1]

                pred_ids = greedy_decode(
                    model=model,
                    src=single_src,
                    src_mask=single_mask,
                    max_len=CONFIG["max_decode_len"],
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )[0].tolist()

                pred_sentence = ids_to_sentence(pred_ids, inv_tgt_vocab)
                ref_sentence = ids_to_sentence(tgt[i].tolist(), inv_tgt_vocab)

                predictions.append(pred_sentence)
                references.append([ref_sentence])

    if evaluate is not None:
        try:
            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=predictions, references=references)
            return float(result["bleu"]) * 100.0
        except Exception:
            pass

    return 0.0


# ============================================================
# MODEL / TRAINING
# ============================================================

def make_model(vocab_de: Dict[str, int], vocab_en: Dict[str, int]):
    """
    Create a fresh model and prevent accidental checkpoint auto-loading.
    """
    if hasattr(model_module.Transformer, "_try_autoload_checkpoint"):
        original_autoload = model_module.Transformer._try_autoload_checkpoint
        model_module.Transformer._try_autoload_checkpoint = lambda self: None
    else:
        original_autoload = None

    model = model_module.Transformer(
        src_vocab_size=len(vocab_de),
        tgt_vocab_size=len(vocab_en),
        d_model=CONFIG["d_model"],
        N=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    )

    if original_autoload is not None:
        model_module.Transformer._try_autoload_checkpoint = original_autoload

    return model


def train_one_run(
    run_name: str,
    smoothing: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_de: Dict[str, int],
    vocab_en: Dict[str, int],
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns:
        epoch_rows: per-epoch metrics
        sample_rows: one sample translation row
        summary_rows: final run summary row
    """
    model = make_model(vocab_de, vocab_en).to(device)

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
        smoothing=smoothing,
    ).to(device)

    epoch_rows: List[Dict] = []

    print(f"\n==================================================")
    print(f"Running: {run_name}")
    print(f"Label smoothing: {smoothing}")
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

            src_mask = model_module.make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = model_module.make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

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
        val_bleu = compute_bleu(model, val_loader, model.inv_tgt_vocab, device)
        val_conf = evaluate_prediction_confidence(model, val_loader, device)

        epoch_rows.append({
            "run_name": run_name,
            "epoch": epoch,
            "label_smoothing": smoothing,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_bleu": val_bleu,
            "prediction_confidence": val_conf,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"{run_name} | Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_acc:.4f} | "
            f"val_bleu={val_bleu:.2f} | "
            f"confidence={val_conf:.4f}"
        )

    # sample translation
    sample_src, sample_tgt = next(iter(val_loader))
    sample_src = sample_src.to(device)
    sample_tgt = sample_tgt.to(device)

    src_mask = model_module.make_src_mask(sample_src[0:1], pad_idx=PAD_IDX).to(device)
    pred_ids = greedy_decode(
        model=model,
        src=sample_src[0:1],
        src_mask=src_mask,
        max_len=CONFIG["max_decode_len"],
        start_symbol=SOS_IDX,
        end_symbol=EOS_IDX,
        device=device,
    )[0].tolist()

    inv_de = build_inverse_vocab(vocab_de)
    inv_en = build_inverse_vocab(vocab_en)

    sample_rows = [{
        "run_name": run_name,
        "source_sentence": ids_to_sentence(sample_src[0].tolist(), inv_de),
        "reference_translation": ids_to_sentence(sample_tgt[0].tolist(), inv_en),
        "predicted_translation": ids_to_sentence(pred_ids, inv_en),
    }]

    summary_rows = [{
        "run_name": run_name,
        "label_smoothing": smoothing,
        "final_train_loss": epoch_rows[-1]["train_loss"],
        "final_val_loss": epoch_rows[-1]["val_loss"],
        "final_val_accuracy": epoch_rows[-1]["val_accuracy"],
        "final_val_bleu": epoch_rows[-1]["val_bleu"],
        "final_prediction_confidence": epoch_rows[-1]["prediction_confidence"],
    }]

    return epoch_rows, sample_rows, summary_rows


# ============================================================
# PLOTS
# ============================================================

def save_plots(epoch_smoothed: List[Dict], epoch_unsmoothed: List[Dict]) -> None:
    epochs = [row["epoch"] for row in epoch_smoothed]

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["train_loss"] for r in epoch_smoothed], label="Label Smoothing 0.1 Train Loss")
    plt.plot(epochs, [r["train_loss"] for r in epoch_unsmoothed], label="Label Smoothing 0.0 Train Loss")
    plt.plot(epochs, [r["val_loss"] for r in epoch_smoothed], label="Label Smoothing 0.1 Val Loss")
    plt.plot(epochs, [r["val_loss"] for r in epoch_unsmoothed], label="Label Smoothing 0.0 Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Label Smoothing Comparison: Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "loss_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["val_accuracy"] for r in epoch_smoothed], label="Label Smoothing 0.1 Val Accuracy")
    plt.plot(epochs, [r["val_accuracy"] for r in epoch_unsmoothed], label="Label Smoothing 0.0 Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Label Smoothing Comparison: Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "val_accuracy_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Validation BLEU
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["val_bleu"] for r in epoch_smoothed], label="Label Smoothing 0.1 Val BLEU")
    plt.plot(epochs, [r["val_bleu"] for r in epoch_unsmoothed], label="Label Smoothing 0.0 Val BLEU")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("Label Smoothing Comparison: Validation BLEU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "val_bleu_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Prediction confidence
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["prediction_confidence"] for r in epoch_smoothed], label="Label Smoothing 0.1 Confidence")
    plt.plot(epochs, [r["prediction_confidence"] for r in epoch_unsmoothed], label="Label Smoothing 0.0 Confidence")
    plt.xlabel("Epoch")
    plt.ylabel("Prediction Confidence")
    plt.title("Label Smoothing Comparison: Prediction Confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "prediction_confidence_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, vocab_de, vocab_en = build_loaders()

    # Run 1: label smoothing = 0.1
    smooth_epochs, smooth_sample, smooth_summary = train_one_run(
        run_name="label_smoothing_0_1",
        smoothing=0.1,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_de=vocab_de,
        vocab_en=vocab_en,
        device=device,
    )

    # Run 2: label smoothing = 0.0
    nosmooth_epochs, nosmooth_sample, nosmooth_summary = train_one_run(
        run_name="label_smoothing_0_0",
        smoothing=0.0,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_de=vocab_de,
        vocab_en=vocab_en,
        device=device,
    )

    # Save per-run CSVs
    save_csv(
        os.path.join(CONFIG["output_dir"], "label_smoothing_0_1_epoch_metrics.csv"),
        smooth_epochs,
        fieldnames=[
            "run_name",
            "epoch",
            "label_smoothing",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_bleu",
            "prediction_confidence",
            "learning_rate",
        ],
    )
    save_csv(
        os.path.join(CONFIG["output_dir"], "label_smoothing_0_0_epoch_metrics.csv"),
        nosmooth_epochs,
        fieldnames=[
            "run_name",
            "epoch",
            "label_smoothing",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_bleu",
            "prediction_confidence",
            "learning_rate",
        ],
    )

    # Save combined CSV
    combined_rows = []
    for row in smooth_epochs:
        combined_rows.append({"setting": "label_smoothing_0_1", **row})
    for row in nosmooth_epochs:
        combined_rows.append({"setting": "label_smoothing_0_0", **row})

    save_csv(
        os.path.join(CONFIG["output_dir"], "combined_epoch_metrics.csv"),
        combined_rows,
        fieldnames=[
            "setting",
            "run_name",
            "epoch",
            "label_smoothing",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_bleu",
            "prediction_confidence",
            "learning_rate",
        ],
    )

    # Save sample translations
    save_csv(
        os.path.join(CONFIG["output_dir"], "label_smoothing_0_1_sample_translation.csv"),
        smooth_sample,
        fieldnames=["run_name", "source_sentence", "reference_translation", "predicted_translation"],
    )
    save_csv(
        os.path.join(CONFIG["output_dir"], "label_smoothing_0_0_sample_translation.csv"),
        nosmooth_sample,
        fieldnames=["run_name", "source_sentence", "reference_translation", "predicted_translation"],
    )

    # Save summary CSV
    summary_rows = smooth_summary + nosmooth_summary
    save_csv(
        os.path.join(CONFIG["output_dir"], "final_summary.csv"),
        summary_rows,
        fieldnames=[
            "run_name",
            "label_smoothing",
            "final_train_loss",
            "final_val_loss",
            "final_val_accuracy",
            "final_val_bleu",
            "final_prediction_confidence",
        ],
    )

    # Save plots
    save_plots(smooth_epochs, nosmooth_epochs)

    print("\nDone.")
    print(f"All CSV and PNG files are saved in: {CONFIG['output_dir']}")
    print("Important files created:")
    print(f"- {os.path.join(CONFIG['output_dir'], 'label_smoothing_0_1_epoch_metrics.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'label_smoothing_0_0_epoch_metrics.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'combined_epoch_metrics.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'final_summary.csv')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'loss_comparison.png')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'val_accuracy_comparison.png')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'val_bleu_comparison.png')}")
    print(f"- {os.path.join(CONFIG['output_dir'], 'prediction_confidence_comparison.png')}")


if __name__ == "__main__":
    main()
    