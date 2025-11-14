import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

INPUT_FILE = "random_bubble_bins.csv" # name of csv
TITLE = "2% Labrasol unloaded" # title of graph
XLABEL = "Bin Center Diameter (um)"
YLABEL = "Count"
BIN_WIDTH = 50 
D_MIN, D_MAX = 1, 900 
YMIN = 0.1
OUT_PNG = f'{TITLE.replace(' ', '_').lower()}_plot.png'
COLUMN = 'diameter_um'  # name of column associated with diameter
TIME_GROUPS = [0, 5, 15]
COLOR_MAP = {0: "black", 5: "red", 15: "blue"}

def main():
    # reading dataframe, dropping null values
    df = pd.read_csv(INPUT_FILE)
    diameters = pd.to_numeric(df[COLUMN], errors="coerce").dropna().to_numpy()
    t = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy()

    # binning
    edges = np.arange(0, D_MAX + BIN_WIDTH, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2.0

    # creating histogram
    counts_per_bin = []
    errors_by_bin = []
    for times in TIME_GROUPS:
        counts, _ = np.histogram(diameters[t == times], bins=edges)
        counts_per_bin.append(counts)
        errors_by_bin.append(np.sqrt(counts))

    counts_per_bin = np.array(counts_per_bin) 
    errors_by_bin = np.array(errors_by_bin)
    
    n_groups = len(TIME_GROUPS)
    total_width = 0.78
    bar_w = (total_width / n_groups) * BIN_WIDTH

    # plotting
    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    # bars + errors
    for i, lvl in enumerate(TIME_GROUPS):
        offsets = (i - (n_groups - 1) / 2.0) * bar_w
        x = centers + offsets
        y = counts_per_bin[i]
        e = errors_by_bin[i]

        ax.bar(
            x,
            y,
            width=bar_w,
            color=COLOR_MAP[lvl],
            edgecolor="black",
            linewidth=0.6,
            align="center",
            zorder=2,
        )
        ax.errorbar(
            x, y, yerr=e,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=3,
        )

    # axes formatting
    ax.set_title(TITLE, pad=6)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, 100)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(['0.1', '1', '10', '100'])

    # x ticks
    ax.set_xlim(0, D_MAX)  # keep the full diameter range
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # only major labels on graph
    ax.tick_params(axis="x", which="major", labelsize=8, length=6)
    ax.tick_params(axis="x", which="minor", labelsize=0, length=3)  # unlabeled minor ticks
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[lvl], edgecolor="black")
               for lvl in TIME_GROUPS]
    labels = [str(lvl) for lvl in TIME_GROUPS]
    ax.legend(handles, labels, loc="upper right", fontsize=8,
              frameon=True, facecolor="white", edgecolor="black")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
