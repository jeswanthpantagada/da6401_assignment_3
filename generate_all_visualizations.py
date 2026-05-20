import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(".")
OUT_DIR = BASE_DIR / "final_visualizations"
OUT_DIR.mkdir(exist_ok=True)

# Input folders from your 5 experiments
NOAM_DIR = BASE_DIR / "csv_outputs"
SCALING_DIR = BASE_DIR / "scaling_factor_outputs"
ATTN_DIR = BASE_DIR / "attention_outputs"
POS_DIR = BASE_DIR / "positional_outputs"
LS_DIR = BASE_DIR / "label_smoothing_outputs"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def get_last_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]


def make_results_table(rows: List[Dict], out_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------
# 2.1 Noam vs Fixed LR
# ---------------------------------------------------------------------
def plot_noam_vs_fixed_lr() -> pd.DataFrame:
    overlay_path = NOAM_DIR / "overlay_comparison.csv"
    df = load_csv(overlay_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["noam_train_loss"], label="Noam Scheduler")
    plt.plot(df["epoch"], df["fixed_train_loss"], label="Fixed LR")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("2.1 Training Loss: Noam vs Fixed LR")
    plt.legend()
    save_plot(OUT_DIR / "noam_vs_fixedlr_training_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["noam_val_accuracy"], label="Noam Scheduler")
    plt.plot(df["epoch"], df["fixed_val_accuracy"], label="Fixed LR")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("2.1 Validation Accuracy: Noam vs Fixed LR")
    plt.legend()
    save_plot(OUT_DIR / "noam_vs_fixedlr_validation_accuracy.png")

    noam_df = load_csv(NOAM_DIR / "noam_scheduler.csv")
    fixed_df = load_csv(NOAM_DIR / "fixed_lr.csv")

    summary = pd.DataFrame([
        {
            "experiment": "2.1 Noam Scheduler",
            "final_train_loss": float(get_last_row(noam_df)["train_loss"]),
            "final_val_accuracy": float(get_last_row(noam_df)["val_accuracy"]),
        },
        {
            "experiment": "2.1 Fixed LR",
            "final_train_loss": float(get_last_row(fixed_df)["train_loss"]),
            "final_val_accuracy": float(get_last_row(fixed_df)["val_accuracy"]),
        },
    ])
    summary.to_csv(OUT_DIR / "noam_vs_fixedlr_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# 2.2 Scaling Factor
# ---------------------------------------------------------------------
def plot_scaling_factor() -> pd.DataFrame:
    overlay_path = SCALING_DIR / "overlay_comparison.csv"
    grad_path = SCALING_DIR / "gradient_norms_first_1000_steps.csv"

    df = load_csv(overlay_path)
    grad_df = load_csv(grad_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["with_scaling_train_loss"], label="With scaling 1/sqrt(dk)")
    plt.plot(df["epoch"], df["without_scaling_train_loss"], label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("2.2 Training Loss: Scaling vs No Scaling")
    plt.legend()
    save_plot(OUT_DIR / "scaling_factor_training_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["with_scaling_val_accuracy"], label="With scaling 1/sqrt(dk)")
    plt.plot(df["epoch"], df["without_scaling_val_accuracy"], label="Without scaling")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("2.2 Validation Accuracy: Scaling vs No Scaling")
    plt.legend()
    save_plot(OUT_DIR / "scaling_factor_validation_accuracy.png")

    for col, fname, title in [
        ("grad_norm_q", "scaling_factor_grad_norm_q.png", "2.2 Query Gradient Norm (First 1000 Steps)"),
        ("grad_norm_k", "scaling_factor_grad_norm_k.png", "2.2 Key Gradient Norm (First 1000 Steps)"),
    ]:
        plt.figure(figsize=(10, 5))
        for run_name in sorted(grad_df["run_name"].unique()):
            sub = grad_df[grad_df["run_name"] == run_name].sort_values("step")
            plt.plot(sub["step"], sub[col], label=run_name)
        plt.xlabel("Step")
        plt.ylabel("Gradient Norm")
        plt.title(title)
        plt.legend()
        save_plot(OUT_DIR / fname)

    summary = pd.DataFrame([
        {
            "experiment": "2.2 With Scaling",
            "final_train_loss": float(get_last_row(df.assign(_setting="with")[["epoch", "with_scaling_train_loss", "with_scaling_val_accuracy"]]).get("with_scaling_train_loss")),
            "final_val_accuracy": float(get_last_row(df)["with_scaling_val_accuracy"]),
        },
        {
            "experiment": "2.2 Without Scaling",
            "final_train_loss": float(get_last_row(df)["without_scaling_train_loss"]),
            "final_val_accuracy": float(get_last_row(df)["without_scaling_val_accuracy"]),
        },
    ])
    summary.to_csv(OUT_DIR / "scaling_factor_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# 2.3 Attention Visualization
# ---------------------------------------------------------------------
def reconstruct_attention_matrix(attn_long: pd.DataFrame, head: int) -> Tuple[pd.DataFrame, List[str]]:
    sub = attn_long[attn_long["head"] == head].copy()
    sub = sub.sort_values(["query_index", "key_index"])

    token_order = (
        sub.sort_values("query_index")
        .drop_duplicates("query_index")[["query_index", "query_token"]]
        .sort_values("query_index")
    )
    tokens = token_order["query_token"].tolist()

    matrix = sub.pivot_table(
        index="query_index",
        columns="key_index",
        values="weight",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    matrix.index = tokens
    matrix.columns = tokens
    return matrix, tokens


def plot_attention_from_csv() -> pd.DataFrame:
    long_csv = ATTN_DIR / "last_encoder_attention_long.csv"
    summary_csv = ATTN_DIR / "last_encoder_attention_summary.csv"

    if not long_csv.exists():
        return pd.DataFrame()

    attn_long = load_csv(long_csv)
    attn_summary = safe_load_csv(summary_csv)

    # Individual heads
    for head in sorted(attn_long["head"].unique()):
        matrix, tokens = reconstruct_attention_matrix(attn_long, int(head))
        plt.figure(figsize=(9, 8))
        plt.imshow(matrix.values, aspect="auto", interpolation="nearest")
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
        plt.yticks(range(len(tokens)), tokens, fontsize=7)
        plt.colorbar()
        plt.title(f"2.3 Last Encoder Layer Attention - Head {head}")
        save_plot(OUT_DIR / f"attention_head_{head}.png")

    # Grid plot
    heads = sorted(attn_long["head"].unique())
    n_heads = len(heads)
    cols = 2
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i, head in enumerate(heads):
        matrix, tokens = reconstruct_attention_matrix(attn_long, int(head))
        ax = axes[i]
        im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Head {head}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(heads), len(axes)):
        axes[j].axis("off")

    plt.suptitle("2.3 Attention Heads Grid", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "attention_heads_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

    if attn_summary is not None and not attn_summary.empty:
        # Summary plots from CSV
        for col, fname, title in [
            ("mean_entropy", "attention_mean_entropy.png", "2.3 Mean Attention Entropy per Head"),
            ("mean_diagonal_mass", "attention_diagonal_mass.png", "2.3 Mean Diagonal Attention Mass per Head"),
            ("mean_attention_distance", "attention_distance.png", "2.3 Mean Attention Distance per Head"),
        ]:
            plt.figure(figsize=(10, 5))
            plt.bar(attn_summary["head"].astype(str), attn_summary[col])
            plt.xlabel("Head")
            plt.ylabel(col)
            plt.title(title)
            save_plot(OUT_DIR / fname)

        attn_summary.to_csv(OUT_DIR / "attention_summary_from_csv.csv", index=False)

    return pd.DataFrame({
        "experiment": ["2.3 Attention Visualization"],
        "note": ["Attention heatmaps generated from last encoder layer"],
    })


# ---------------------------------------------------------------------
# 2.4 Positional Encoding
# ---------------------------------------------------------------------
def plot_positional_encoding() -> pd.DataFrame:
    df = load_csv(POS_DIR / "combined_epoch_metrics.csv")

    sin_df = df[df["setting"] == "sinusoidal"].sort_values("epoch")
    learned_df = df[df["setting"] == "learned"].sort_values("epoch")

    plt.figure(figsize=(10, 5))
    plt.plot(sin_df["epoch"], sin_df["train_loss"], label="Sinusoidal Train Loss")
    plt.plot(learned_df["epoch"], learned_df["train_loss"], label="Learned Train Loss")
    plt.plot(sin_df["epoch"], sin_df["val_loss"], label="Sinusoidal Val Loss")
    plt.plot(learned_df["epoch"], learned_df["val_loss"], label="Learned Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("2.4 Positional Encoding: Loss Comparison")
    plt.legend()
    save_plot(OUT_DIR / "positional_loss_comparison.png")

    plt.figure(figsize=(10, 5))
    plt.plot(sin_df["epoch"], sin_df["val_accuracy"], label="Sinusoidal Val Accuracy")
    plt.plot(learned_df["epoch"], learned_df["val_accuracy"], label="Learned Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("2.4 Positional Encoding: Validation Accuracy")
    plt.legend()
    save_plot(OUT_DIR / "positional_accuracy_comparison.png")

    plt.figure(figsize=(10, 5))
    plt.plot(sin_df["epoch"], sin_df["val_bleu"], label="Sinusoidal Val BLEU")
    plt.plot(learned_df["epoch"], learned_df["val_bleu"], label="Learned Val BLEU")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("2.4 Positional Encoding: Validation BLEU")
    plt.legend()
    save_plot(OUT_DIR / "positional_bleu_comparison.png")

    summary = pd.DataFrame([
        {
            "experiment": "2.4 Sinusoidal",
            "final_val_accuracy": float(sin_df.iloc[-1]["val_accuracy"]),
            "final_val_bleu": float(sin_df.iloc[-1]["val_bleu"]),
        },
        {
            "experiment": "2.4 Learned",
            "final_val_accuracy": float(learned_df.iloc[-1]["val_accuracy"]),
            "final_val_bleu": float(learned_df.iloc[-1]["val_bleu"]),
        },
    ])
    summary.to_csv(OUT_DIR / "positional_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# 2.5 Label Smoothing
# ---------------------------------------------------------------------
def plot_label_smoothing() -> pd.DataFrame:
    df = load_csv(LS_DIR / "combined_epoch_metrics.csv")

    smooth_df = df[df["setting"] == "label_smoothing_0_1"].sort_values("epoch")
    nosmooth_df = df[df["setting"] == "label_smoothing_0_0"].sort_values("epoch")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_df["epoch"], smooth_df["train_loss"], label="Smoothing 0.1 Train Loss")
    plt.plot(nosmooth_df["epoch"], nosmooth_df["train_loss"], label="Smoothing 0.0 Train Loss")
    plt.plot(smooth_df["epoch"], smooth_df["val_loss"], label="Smoothing 0.1 Val Loss")
    plt.plot(nosmooth_df["epoch"], nosmooth_df["val_loss"], label="Smoothing 0.0 Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("2.5 Label Smoothing: Loss Comparison")
    plt.legend()
    save_plot(OUT_DIR / "label_smoothing_loss_comparison.png")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_df["epoch"], smooth_df["val_accuracy"], label="Smoothing 0.1 Val Accuracy")
    plt.plot(nosmooth_df["epoch"], nosmooth_df["val_accuracy"], label="Smoothing 0.0 Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("2.5 Label Smoothing: Validation Accuracy")
    plt.legend()
    save_plot(OUT_DIR / "label_smoothing_accuracy_comparison.png")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_df["epoch"], smooth_df["val_bleu"], label="Smoothing 0.1 Val BLEU")
    plt.plot(nosmooth_df["epoch"], nosmooth_df["val_bleu"], label="Smoothing 0.0 Val BLEU")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("2.5 Label Smoothing: Validation BLEU")
    plt.legend()
    save_plot(OUT_DIR / "label_smoothing_bleu_comparison.png")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_df["epoch"], smooth_df["prediction_confidence"], label="Smoothing 0.1 Confidence")
    plt.plot(nosmooth_df["epoch"], nosmooth_df["prediction_confidence"], label="Smoothing 0.0 Confidence")
    plt.xlabel("Epoch")
    plt.ylabel("Prediction Confidence")
    plt.title("2.5 Label Smoothing: Prediction Confidence")
    plt.legend()
    save_plot(OUT_DIR / "label_smoothing_confidence_comparison.png")

    summary = pd.DataFrame([
        {
            "experiment": "2.5 Label Smoothing 0.1",
            "final_val_accuracy": float(smooth_df.iloc[-1]["val_accuracy"]),
            "final_val_bleu": float(smooth_df.iloc[-1]["val_bleu"]),
            "final_prediction_confidence": float(smooth_df.iloc[-1]["prediction_confidence"]),
        },
        {
            "experiment": "2.5 Label Smoothing 0.0",
            "final_val_accuracy": float(nosmooth_df.iloc[-1]["val_accuracy"]),
            "final_val_bleu": float(nosmooth_df.iloc[-1]["val_bleu"]),
            "final_prediction_confidence": float(nosmooth_df.iloc[-1]["prediction_confidence"]),
        },
    ])
    summary.to_csv(OUT_DIR / "label_smoothing_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# Final master summary
# ---------------------------------------------------------------------
def build_master_summary(all_summaries: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in all_summaries:
        if df is None or df.empty:
            continue
        rows.extend(df.to_dict(orient="records"))
    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "master_summary.csv", index=False)
    return master


def main() -> None:
    summaries: List[pd.DataFrame] = []

    # Each block runs only if its inputs exist
    try:
        summaries.append(plot_noam_vs_fixed_lr())
    except Exception as e:
        print(f"[WARN] 2.1 skipped: {e}")

    try:
        summaries.append(plot_scaling_factor())
    except Exception as e:
        print(f"[WARN] 2.2 skipped: {e}")

    try:
        attn_summary = plot_attention_from_csv()
        if attn_summary is not None and not attn_summary.empty:
            summaries.append(attn_summary)
    except Exception as e:
        print(f"[WARN] 2.3 skipped: {e}")

    try:
        summaries.append(plot_positional_encoding())
    except Exception as e:
        print(f"[WARN] 2.4 skipped: {e}")

    try:
        summaries.append(plot_label_smoothing())
    except Exception as e:
        print(f"[WARN] 2.5 skipped: {e}")

    master = build_master_summary(summaries)

    print("\nDone.")
    print(f"All final graphs saved in: {OUT_DIR}")
    if not master.empty:
        print(f"Master summary saved to: {OUT_DIR / 'master_summary.csv'}")


if __name__ == "__main__":
    main()