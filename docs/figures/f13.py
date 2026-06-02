import torch
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

groups = [
    "wo/ point+match loss",
    "proposed",
    "wo/ register attention",
    "w/ point head",
    "w/ more 10% self-supervision",
]
vals = [0.078, 0.073, 0.071, 0.070, 0.070]
colors = ["#9aa6b1", "#c89bb8", "#6fb3a4", "#c6b89c", "#b2a4c8"]

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)

x = torch.arange(len(groups)).numpy()
width = 0.7

bars = ax.bar(x, vals, width, color=colors, edgecolor="white")

for bar, v in zip(bars, vals):
    ax.annotate(
        f"{v:.3f}",
        (bar.get_x() + bar.get_width() / 2, v),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=22,
        fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=18, fontweight="bold")
ax.set_ylabel("point error", fontsize=22, fontweight="bold")
ax.tick_params(axis="y", labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_ylim(0, 0.090)

plt.tight_layout()
plt.savefig(__file__.rsplit(".", 1)[0] + ".png")
plt.close()
