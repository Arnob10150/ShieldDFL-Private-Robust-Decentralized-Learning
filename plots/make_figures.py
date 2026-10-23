from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "Centralized MLP": "#3b5bdb",
    "Local-only": "#868e96",
    "FedAvg": "#1971c2",
    "FedProx": "#0ca678",
    "P2P Gossip FL": "#f08c00",
    "ShieldDFL": "#c2255c",
    "ShieldDFL+DP": "#9c36b5",
    "Median": "#2b8a3e",
    "Trimmed Mean": "#5c940d",
}


def ensure_figures(results_dir: Path) -> Path:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save(fig, fig_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}.png", dpi=300)
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)


def read_csv(results_dir: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / rel)


def draw_box(ax, xy, text, width=0.18, height=0.12, color="#edf2ff"):
    rect = plt.Rectangle(xy, width, height, facecolor=color, edgecolor="#343a40", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)


def architecture(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [(0.12, 0.66), (0.42, 0.76), (0.72, 0.66), (0.72, 0.22), (0.42, 0.12), (0.12, 0.22)]
    labels = ["Hospital\nedge", "Factory\nedge", "Campus\nedge", "SDN\ngateway", "ISP\nedge", "IoT\nzone"]
    for xy, label in zip(nodes, labels):
        draw_box(ax, xy, label, color="#e7f5ff")
    for i, src in enumerate(nodes):
        dst = nodes[(i + 1) % len(nodes)]
        ax.annotate("", xy=(dst[0] + 0.09, dst[1] + 0.06), xytext=(src[0] + 0.09, src[1] + 0.06),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#495057"))
    draw_box(ax, (0.39, 0.43), "Trust filter\n+ DP log\n+ gossip update", width=0.24, height=0.16, color="#fff3bf")
    ax.text(0.5, 0.035, "Raw traffic remains local; only clipped/noised model updates and trust metadata are exchanged.",
            ha="center", va="bottom", fontsize=8)
    save(fig, fig_dir, "architecture")


def workflow(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = ["CICDDoS2019\nflows", "Clean +\nscale", "Dirichlet\nclients", "Local\ntraining", "Trust-aware\ngossip", "Metrics +\nfigures"]
    xs = [0.03, 0.20, 0.37, 0.54, 0.71, 0.88]
    for x, step in zip(xs, steps):
        draw_box(ax, (x, 0.42), step, width=0.12, height=0.20, color="#f8f9fa")
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2, 0.52), xytext=(x1 + 0.12, 0.52),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#495057"))
    save(fig, fig_dir, "workflow")


def main_comparison(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/main_results.csv")
    plot_df = df.melt(id_vars=["method"], value_vars=["accuracy", "f1_macro", "auc"],
                      var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    sns.barplot(data=plot_df, y="method", x="score", hue="metric", ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(title="", fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, fig_dir, "main_comparison")


def privacy(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/privacy_results.csv").copy()
    def label_eps(value):
        text = str(value)
        if text.lower() == "inf":
            return "No DP"
        try:
            number = float(text)
            return f"$\\epsilon={number:g}$"
        except ValueError:
            return text

    df["epsilon_sort"] = df["epsilon"].map(lambda value: 999 if str(value).lower() == "inf" else float(value))
    df["epsilon_label"] = df["epsilon"].map(label_eps)
    df = df.sort_values("epsilon_sort")
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(df["epsilon_label"], df["f1_macro"], marker="o", markersize=7, linewidth=2.2, label="Macro-F1", color="#c2255c")
    ax.plot(df["epsilon_label"], df["auc"], marker="s", markersize=7, linewidth=2.2, label="AUC", color="#1971c2")
    ax.set_xlabel("Privacy budget", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    save(fig, fig_dir, "privacy_utility")


def poisoning(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/poisoning_results.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for method, group in df.groupby("method"):
        ax.plot(group["malicious_ratio"] * 100, group["f1_macro"], marker="o", markersize=6, linewidth=2.0, label=method,
                color=PALETTE.get(method))
    ax.set_xlabel("Malicious clients (%)", fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(ncol=2, fontsize=9)
    save(fig, fig_dir, "poisoning_robustness")


def convergence(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/round_metrics.csv")
    df = df[(df["malicious_ratio"] == 0.0) & (df["attack_type"] == "none")]
    keep = ["FedAvg", "FedProx", "P2P Gossip FL", "ShieldDFL"]
    df = df[df["method"].isin(keep)]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for method, group in df.groupby("method"):
        ax.plot(group["round"], group["f1_macro"], marker="o", markersize=5, linewidth=2.0, label=method, color=PALETTE.get(method))
    ax.set_xlabel("Communication round", fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)
    save(fig, fig_dir, "convergence")


def communication(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/main_results.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    sns.barplot(data=df, y="method", x="communication_mb", ax=ax, color="#74c0fc")
    ax.set_xlabel("Total communication (MB)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    save(fig, fig_dir, "communication_cost")


def ablation(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "logs/ablation_results.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    sns.barplot(data=df, y="method", x="f1_macro", ax=ax, color="#f783ac")
    ax.set_xlabel("Macro-F1 under 10% mixed attack", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="both", labelsize=10)
    save(fig, fig_dir, "ablation")


def client_distribution(results_dir: Path, fig_dir: Path) -> None:
    df = read_csv(results_dir, "tables/client_distribution.csv")
    labels = [f"C{int(client_id)}" for client_id in df["client_id"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.barh(labels, df["benign"], label="Benign", color="#4dabf7")
    ax.barh(labels, df["ddos"], left=df["benign"], label="DDoS", color="#fa5252")
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel("Client", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, fig_dir, "client_distribution")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    fig_dir = ensure_figures(results_dir)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    architecture(fig_dir)
    workflow(fig_dir)
    main_comparison(results_dir, fig_dir)
    privacy(results_dir, fig_dir)
    poisoning(results_dir, fig_dir)
    convergence(results_dir, fig_dir)
    communication(results_dir, fig_dir)
    ablation(results_dir, fig_dir)
    client_distribution(results_dir, fig_dir)


if __name__ == "__main__":
    main()
