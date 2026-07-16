"""
make_fig_fold10_forest.py -- clean regeneration of reports/figures/fold10_forest_ap.png,
the held-out test-fold forest plot. Fixes the overlay artifact and relabels M1 as
"QRS-onset-morphology (M1)". Values are the frozen fold-10 results (14 WPW). M6 is not shown:
it is PTB-only (7 held-out positives) and not comparable to the 14-positive figures here.
Read-only on data; writes only the PNG (a backup of the current file is kept).

Run:  python make_fig_fold10_forest.py
"""
import os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(ROOT, "reports", "figures", "fold10_forest_ap.png")

NAVY = "#20304a"
RED  = "#b31b1b"
GREY = "#5b6472"

# frozen fold-10 results (AP, CI low, CI high), top -> bottom, as displayed
ROWS = [
    ("Convolutional network (M7)",                0.74, 0.53, 0.94, NAVY, False),
    ("Wavelet + median-beat ensemble (selected)", 0.59, 0.35, 0.85, RED,  True),
    ("Feature-union model",                       0.55, 0.31, 0.81, NAVY, False),
    ("Median-beat morphology (M4)",               0.55, 0.30, 0.83, NAVY, False),
    ("Wavelet-localization (M3)",                 0.54, 0.31, 0.80, NAVY, False),
    ("Spatial-VCG (M5)",                          0.30, 0.10, 0.59, NAVY, False),
    ("Global-statistical (M2)",                   0.28, 0.08, 0.54, NAVY, False),
    ("QRS-onset-morphology (M1)",                 0.25, 0.07, 0.50, NAVY, False),
]
ENS_AP = 0.59

n  = len(ROWS)
ys = list(range(n, 0, -1))                      # top row highest y

fig, ax = plt.subplots(figsize=(14.5, 5.4))
ax.axvline(ENS_AP, color="#9aa3af", ls="--", lw=1.3, zorder=1,
           label="selected ensemble AP = %.2f" % ENS_AP)

for y, (name, ap, lo, hi, col, sel) in zip(ys, ROWS):
    ax.plot([lo, hi], [y, y], color=col, lw=3.0 if sel else 2.2, zorder=2,
            solid_capstyle="round")
    ax.scatter([ap], [y], s=180 if sel else 95, color=col, zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.text(1.03, y, "%.2f  [%.2f, %.2f]" % (ap, lo, hi), va="center", ha="left",
            fontsize=12.5, color=col, fontweight="bold" if sel else "normal", clip_on=False)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=12.5)
for lab, r in zip(ax.get_yticklabels(), ROWS):
    if r[5]:                                     # selected row
        lab.set_color(RED); lab.set_fontweight("bold")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.4, n + 0.9)
ax.set_xlabel("Held-out test-fold Average Precision (95% CI)", fontsize=13)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis="x", labelsize=11.5)
ax.set_title("All 95% CIs overlap: no representation separates at 14 test WPW",
             fontsize=13, style="italic", color=GREY, pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.18)
ax.legend(loc="lower right", fontsize=11, frameon=False)

fig.subplots_adjust(left=0.31, right=0.86, top=0.90, bottom=0.13)

if os.path.exists(FIG):
    bak = FIG.replace(".png", ".preclean.png")
    if not os.path.exists(bak):
        shutil.copy(FIG, bak)
fig.savefig(FIG, dpi=200, facecolor="white")
print("wrote", FIG)
