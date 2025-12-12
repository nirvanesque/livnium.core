import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def format_value(val):
    """Format numbers nicely for labels."""
    if isinstance(val, int):
        return f"{val:,}"
    return f"{val:,.1f}"


def main():
    # Data
    models_size = ["RoBERTa-Base", "BERT-Base", "DeBERTa-v3", "Livnium (Ours)"]
    size_mb = [499, 440, 371, 52.3]
    colors_size = ["#444444", "#444444", "#444444", "#00E5FF"]

    models_speed = ["Livnium (Ours)", "DistilBERT", "BERT-Base"]
    speed_samples = [7384, 715, 380]
    colors_speed = ["#00E5FF", "#444444", "#444444"]

    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#121212")
    ax1.set_facecolor("#121212")
    ax2.set_facecolor("#121212")

    fig.suptitle(
        "Livnium Efficiency vs. Transformer Baselines",
        fontsize=20,
        fontweight="bold",
        color="white",
        y=1.05,
    )

    # Chart 1: Storage
    y_pos_size = np.arange(len(models_size))
    rects1 = ax1.barh(y_pos_size, size_mb, color=colors_size)
    ax1.set_yticks(y_pos_size)
    ax1.set_yticklabels(models_size, fontsize=12, color="white")
    ax1.invert_yaxis()
    ax1.set_xlabel("Model Size (MB)", fontsize=12, color="white")
    ax1.set_title("Storage Footprint (Lower is Better)", fontsize=14, color="white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_color("#555555")
    ax1.spines["left"].set_color("#555555")
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")

    for i, rect in enumerate(rects1):
        width = rect.get_width()
        text_color = "#00E5FF" if models_size[i] == "Livnium (Ours)" else "white"
        ax1.text(
            width + 10,
            rect.get_y() + rect.get_height() / 2,
            f"{format_value(size_mb[i])} MB",
            va="center",
            fontsize=12,
            color=text_color,
            fontweight="bold",
        )

    # Chart 2: Speed
    x_pos_speed = np.arange(len(models_speed))
    rects2 = ax2.bar(x_pos_speed, speed_samples, color=colors_speed)
    ax2.set_xticks(x_pos_speed)
    ax2.set_xticklabels(models_speed, fontsize=12, color="white")
    ax2.set_ylabel("Samples/Sec", fontsize=12, color="white")
    ax2.set_title("Inference Speed (Higher is Better)", fontsize=14, color="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#555555")
    ax2.spines["left"].set_color("#555555")
    ax2.tick_params(axis="x", colors="white")
    ax2.tick_params(axis="y", colors="white")

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        text_color = "#00E5FF" if models_speed[i] == "Livnium (Ours)" else "white"
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            height + 100,
            format_value(height),
            ha="center",
            va="bottom",
            fontsize=12,
            color=text_color,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        "livnium_efficiency_dark.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="#121212",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
