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

delta = torch.tensor(
    [
        [92.4, 98.4, 95.8, 93.3, 71.9, 85.0],
        [92.9, 98.7, 96.3, 97.0, 72.5, 93.1],
        [93.8, 96.2, 94.8, 97.4, 74.1, 92.9],
        [91.9, 99.1, 97.4, 95.2, 79.2, 92.2],
        [92.8, 99.2, 99.6, 97.4, 82.5, 95.5],
        [93.0, 99.5, 99.6, 97.7, 86.1, 94.3],
        [94.6, 99.6, 99.8, 98.4, 89.5, 97.4],
        [96.3, 99.7, 99.8, 98.7, 93.5, 98.3],
    ]
)

absrel = torch.tensor(
    [
        [0.075, 0.030, 0.056, 0.068, 0.263, 0.148],
        [0.070, 0.022, 0.035, 0.049, 0.251, 0.052],
        [0.065, 0.057, 0.083, 0.042, 0.207, 0.083],
        [0.073, 0.019, 0.036, 0.055, 0.189, 0.064],
        [0.068, 0.011, 0.016, 0.041, 0.144, 0.046],
        [0.063, 0.010, 0.015, 0.039, 0.118, 0.049],
        [0.058, 0.010, 0.012, 0.038, 0.097, 0.041],
        [0.050, 0.007, 0.009, 0.030, 0.081, 0.035],
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=150)

ax = axes[0]
ax.imshow(
    per_column_normalize(delta, descending=False).numpy(),
    cmap=TINTED_RDYLGN,
    aspect="auto",
    vmin=0,
    vmax=1,
)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=16, fontweight="bold")
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=16, fontweight="bold")
ax.set_xlabel("δ<1.25", fontsize=20, fontweight="bold")
for i in range(len(methods)):
    for j in range(len(datasets)):
        v = delta[i, j].item()
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

ax = axes[1]
ax.imshow(
    per_column_normalize(absrel, descending=True).numpy(),
    cmap=TINTED_RDYLGN,
    aspect="auto",
    vmin=0,
    vmax=1,
)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=16, fontweight="bold")
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=16, fontweight="bold")
ax.set_xlabel("AbsRel", fontsize=20, fontweight="bold")
for i in range(len(methods)):
    for j in range(len(datasets)):
        v = absrel[i, j].item()
        ax.text(
            j,
            i,
            f"{v:.3f}",
            ha="center",
            va="center",
            color="black",
            fontsize=18,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig(__file__.rsplit(".", 1)[0] + ".png")
plt.close()
