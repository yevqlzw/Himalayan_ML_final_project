import matplotlib.pyplot as plt
from cycler import cycler

from src.config import FIGURES_DIR

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DARK_BLUE = "#14498c"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA]),
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_figure(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=200, bbox_inches="tight")
