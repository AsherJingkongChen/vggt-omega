import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

TINT = 0.4

base = mpl.colormaps["RdYlGn"](torch.linspace(0, 1, 256).numpy())
tinted = base * (1 - TINT) + TINT
tinted[:, 3] = 1.0
TINTED_RDYLGN = LinearSegmentedColormap.from_list("TintedRdYlGn", tinted)


def per_column_normalize(data: torch.Tensor, descending: bool = False) -> torch.Tensor:
    """Per-column min-max normalize to [0, 1] preserving value ratios."""
    col_min = data.min(dim=0, keepdim=True).values
    col_max = data.max(dim=0, keepdim=True).values
    norm = (data - col_min) / (col_max - col_min).clamp(min=1e-12)
    return 1 - norm if descending else norm


methods = [
    "MonST3R",
    "MapAnything",
    "MegaSaM",
    "VGGT",
    "PI3",
    "DA3",
    "Ours-1B",
    "Ours-10B",
]
datasets = ["7 Scenes", "NRGBD", "ETH3D", "DyCheck", "Sintel", "TUM-Dynamic"]

auc3 = torch.tensor(
    [
        [9.0, 13.9, 1.7, 11.5, 4.3, 7.7],
        [5.8, 35.2, 13.2, 6.1, 2.9, 4.3],
        [10.6, 17.2, 5.9, 26.8, 22.5, 15.4],
        [10.9, 81.7, 18.8, 21.0, 15.0, 16.6],
        [13.3, 83.8, 35.3, 23.3, 14.8, 16.1],
        [18.7, 86.4, 46.1, 32.1, 16.2, 20.8],
        [29.6, 89.7, 49.8, 38.4, 35.3, 30.2],
        [36.4, 92.5, 56.3, 43.7, 40.0, 36.4],
    ]
)

auc30 = torch.tensor(
    [
        [68.3, 79.7, 14.3, 45.4, 45.8, 48.5],
        [61.4, 88.9, 51.0, 60.3, 31.6, 40.2],
        [71.8, 83.1, 38.1, 53.1, 58.3, 59.0],
        [74.4, 97.7, 62.1, 78.7, 50.0, 61.2],
        [77.0, 98.2, 79.6, 81.0, 53.5, 59.2],
        [78.2, 98.4, 87.0, 83.9, 52.7, 62.7],
        [83.1, 98.8, 88.5, 87.3, 73.0, 82.3],
        [88.2, 99.1, 90.4, 90.9, 79.1, 87.5],
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=150)

for ax, data, name in [
    (axes[0], auc3, "AUC@3°"),
    (axes[1], auc30, "AUC@30°"),
]:
    q = per_column_normalize(data, descending=False)
    ax.imshow(q.numpy(), cmap=TINTED_RDYLGN, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        datasets, rotation=30, ha="right", fontsize=16, fontweight="bold"
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=16, fontweight="bold")
    ax.set_xlabel(name, fontsize=20, fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(datasets)):
            v = data[i, j].item()
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=18,
                fontweight="bold",
            )

plt.tight_layout()
plt.savefig(__file__.rsplit(".", 1)[0] + ".png")
plt.close()
