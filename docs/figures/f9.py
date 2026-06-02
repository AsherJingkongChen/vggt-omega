import torch
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

benchmarks = [
    "Sintel\n$(S \\approx 46)$",
    "KITTI\n$(S \\approx 264)$",
    "Bonn\n$(S \\approx 623)$",
    "TUM-dynamics\n$(S \\approx 1000)$",
    "7-Scenes\n$(S \\approx 1000)$",
    "NRGBD\n$(S \\approx 1000)$",
    "ScanNet\n$(S \\approx 1000)$",
]
backbones = ["VGGT", "DA3", r"$\pi^3$", r"VGGT-$\Omega$"]
colors = ["#9aa6b1", "#c6b89c", "#c89bb8", "#6fb3a4"]

times = torch.tensor(
    [
        [2.0, 2.6, 1.6, 1.3],
        [31.5, 28.8, 22.6, 16.6],
        [552.4, 493.4, 413.1, 252.8],
        [993.7, 870.8, 739.4, 448.7],
        [1102.7, 976.4, 830.2, 505.3],
        [1110.4, 983.3, 835.7, 508.4],
        [1162.7, 1018.1, 865.1, 525.1],
    ]
)

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)

n_groups = len(benchmarks)
n_bars = len(backbones)
group_width = 0.82
bar_width = group_width / n_bars
x = torch.arange(n_groups).numpy()

for i, (name, color) in enumerate(zip(backbones, colors)):
    offsets = x - group_width / 2 + bar_width * (i + 0.5)
    vals = times[:, i].numpy()
    bars = ax.bar(offsets, vals, bar_width, color=color, label=name, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:.1f}",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=16, fontweight="bold")
ax.set_ylabel("inference time (s)", fontsize=22, fontweight="bold")
ax.tick_params(axis="y", labelsize=16)
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.legend(loc="upper left", fontsize=20)
ax.set_ylim(0.8, 2500)

plt.tight_layout()
plt.savefig(__file__.rsplit(".", 1)[0] + ".png")
plt.close()
