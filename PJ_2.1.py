import os
import csv
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Multi30kDataset, collate_fn, build_vocab_from_train
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 15,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "warmup_steps": 4000,
    "fixed_lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

PAD_IDX = 0

# =========================================================
# SEED
# =========================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# =========================================================
# DATASET
# =========================================================

print("Loading dataset...")

vocab_de, vocab_en = build_vocab_from_train()

train_dataset = Multi30kDataset(
    split="train",
    vocab_de=vocab_de,
    vocab_en=vocab_en,
)

val_dataset = Multi30kDataset(
    split="validation",
    vocab_de=vocab_de,
    vocab_en=vocab_en,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=collate_fn,
)

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# =========================================================
# VALIDATION ACCURACY
# =========================================================

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()

    total_correct = 0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_input).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)

        pred = output.argmax(dim=-1)

        mask = tgt_output != PAD_IDX

        total_correct += ((pred == tgt_output) & mask).sum().item()
        total_tokens += mask.sum().item()

    return total_correct / total_tokens

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_model(use_noam=True):

    device = CONFIG["device"]

    model = Transformer(
        src_vocab_size=len(vocab_de),
        tgt_vocab_size=len(vocab_en),
        d_model=CONFIG["d_model"],
        N=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    ).to(device)

    if use_noam:

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

        experiment_name = "noam_scheduler"

    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["fixed_lr"],
        )

        scheduler = None

        experiment_name = "fixed_lr"

    results = []

    print(f"\nStarting Experiment: {experiment_name}")

    for epoch in range(1, CONFIG["epochs"] + 1):

        model.train()

        total_loss = 0

        pbar = tqdm(train_loader)

        for src, tgt in pbar:

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            optimizer.zero_grad()

            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tgt_output.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            pbar.set_description(
                f"{experiment_name} Epoch {epoch}"
            )

        avg_train_loss = total_loss / len(train_loader)

        val_acc = evaluate_accuracy(
            model,
            val_loader,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        results.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
        })

    return results

# =========================================================
# RUN BOTH EXPERIMENTS
# =========================================================

noam_results = train_model(use_noam=True)

fixed_results = train_model(use_noam=False)

# =========================================================
# SAVE CSV FILES
# =========================================================

os.makedirs("csv_outputs", exist_ok=True)

def save_csv(results, filename):

    path = os.path.join("csv_outputs", filename)

    with open(path, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_accuracy",
                "learning_rate",
            ]
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"Saved CSV: {path}")

save_csv(noam_results, "noam_scheduler.csv")

save_csv(fixed_results, "fixed_lr.csv")

# =========================================================
# COMBINED OVERLAY CSV
# =========================================================

overlay_csv = os.path.join(
    "csv_outputs",
    "overlay_comparison.csv"
)

with open(overlay_csv, "w", newline="") as csvfile:

    fieldnames = [
        "epoch",
        "noam_train_loss",
        "fixed_train_loss",
        "noam_val_accuracy",
        "fixed_val_accuracy",
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for i in range(CONFIG["epochs"]):

        writer.writerow({
            "epoch": i + 1,
            "noam_train_loss": noam_results[i]["train_loss"],
            "fixed_train_loss": fixed_results[i]["train_loss"],
            "noam_val_accuracy": noam_results[i]["val_accuracy"],
            "fixed_val_accuracy": fixed_results[i]["val_accuracy"],
        })

print(f"Saved CSV: {overlay_csv}")

# =========================================================
# SAVE PLOTS
# =========================================================

epochs = list(range(1, CONFIG["epochs"] + 1))

# Training Loss Plot

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    [x["train_loss"] for x in noam_results],
    label="Noam Scheduler"
)

plt.plot(
    epochs,
    [x["train_loss"] for x in fixed_results],
    label="Fixed LR"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()

plt.savefig(
    "csv_outputs/training_loss_comparison.png"
)

# Validation Accuracy Plot

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    [x["val_accuracy"] for x in noam_results],
    label="Noam Scheduler"
)

plt.plot(
    epochs,
    [x["val_accuracy"] for x in fixed_results],
    label="Fixed LR"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()

plt.savefig(
    "csv_outputs/validation_accuracy_comparison.png"
)

print("\nALL FILES SAVED SUCCESSFULLY")
print("Check the folder: csv_outputs/")