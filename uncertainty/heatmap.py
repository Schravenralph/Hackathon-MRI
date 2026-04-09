import io
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_uncertainty_bar_chart(variance: np.ndarray, class_names: list[str]) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 3))
    mean_var = np.mean(variance)
    colors = ["#d32f2f" if v > mean_var else "#4caf50" for v in variance]
    ax.barh(class_names, variance, color=colors)
    ax.set_xlabel("Prediction Variance (Uncertainty)")
    ax.set_title("Per-Class Uncertainty (MC Dropout)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_mc_dropout_visualization(all_probs: np.ndarray, class_names: list[str]) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = list(range(len(class_names)))
    ax.violinplot(
        [all_probs[:, i] for i in positions],
        positions=positions,
        showmeans=True, showmedians=True,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Softmax Probability")
    ax.set_title(f"MC Dropout Distribution ({all_probs.shape[0]} passes)")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
