r"""
离线实验：比一比"随机挑""铺开挑""贝叶斯挑"
=============================================

题库：现有 29 组设计，每一组的真实指标都已经知道。
玩法：从 5 个初始点出发，每一轮挑一个新点，"揭晓"它的真实指标，更新模型，再挑下一个。

这样不用真跑仿真，也能比较不同策略谁先用更少的次数找到好设计。

产出（outputs\figures\）：
    04_三组方法性能与成本对比.png
    05_优化过程曲线.png
    06_主动选点对照.png
    07_误差反馈消融.png        （去掉不确定度项）
    08_独立重复运行统计.png

放哪里：B_Group/src/experiment.py

------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
    cd /d D:\rf\挑战杯\B_Group
    python -m src.experiment

换随机种子数、每轮次数：

    python -m src.experiment --seeds 20 --steps 10

注意：这是离线实验，结果来自"已有数据里挑"，不是真跑了 200 次仿真。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

try:
    from . import config
    from .gp_model import TARGETS, scan_dataset
    from .predict import margin
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from gp_model import TARGETS, scan_dataset  # type: ignore
    from predict import margin  # type: ignore

FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
for _f in FONT_CANDIDATES:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

GREEN = "#2e7d5b"
RED = "#b23a48"
BLUE = "#1f5c73"
ORANGE = "#c9822b"
GREY = "#8d99a3"

LIMITS = {m.key: m.limit_db for m in config.METRICS}
INIT = 5          # 初始给几个点
BETA = 0.6        # UCB 的探索权重
GOOD = 0.60       # 算"成功"的门槛：余量至少 0.60
START_SEED = 12345


def _scores(df: pd.DataFrame) -> np.ndarray:
    """每组设计的真实余量（用真实指标算，不是预测）。"""
    return np.array([margin({k: float(row[k]) for k in TARGETS}) for _, row in df.iterrows()])


def _fit(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """固定超参的 GP，快且稳，够做离线对比用。"""
    ker = (ConstantKernel(1.0, constant_value_bounds="fixed")
           * RBF(1.0, length_scale_bounds="fixed")
           + WhiteKernel(1e-6, noise_level_bounds="fixed"))
    gp = GaussianProcessRegressor(kernel=ker, normalize_y=True, optimizer=None)
    gp.fit(X, y)
    return gp


def _predict_all(X_train, Y_train, X_pool):
    mu = np.zeros((len(X_pool), len(TARGETS)))
    sd = np.zeros_like(mu)
    for j in range(len(TARGETS)):
        gp = _fit(X_train, Y_train[:, j])
        mu[:, j], sd[:, j] = gp.predict(X_pool, return_std=True)
    return mu, sd


def _margin_of(mu_row: np.ndarray) -> float:
    return margin({TARGETS[j]: float(mu_row[j]) for j in range(len(TARGETS))})


def _margin_sd(sd_row: np.ndarray) -> float:
    return float(min(sd_row[j] / abs(LIMITS[TARGETS[j]]) for j in range(len(TARGETS))))


def run_once(df: pd.DataFrame, true_score: np.ndarray, method: str, seed: int,
             steps: int, beta: float = BETA) -> np.ndarray:
    """跑一次，返回每轮结束时的"当前最好真实余量"。"""
    rng = np.random.default_rng(seed)
    n = len(df)
    X_all = df[["c1_pf", "c2_pf"]].to_numpy(float)
    Y_all = df[TARGETS].to_numpy(float)

    # 统一起点：随机 5 组，但排除成绩最好的 3 组，留出"往上找"的空间
    start_rng = np.random.default_rng(START_SEED)
    pool_start = np.argsort(true_score)[: n - 3]
    chosen = list(start_rng.choice(pool_start, size=INIT, replace=False))

    history = [float(true_score[chosen].max())]

    for _ in range(steps):
        remaining = [i for i in range(n) if i not in chosen]
        if not remaining:
            break

        if method == "random":
            pick = int(rng.choice(remaining))
        elif method == "grid":
            # 网格扫描：按 C1 从小到大、C2 从小到大依次试
            order = np.lexsort((X_all[:, 1], X_all[:, 0]))
            pick = next(i for i in order if i in remaining)
        else:  # bayes / bayes_mean
            sc = StandardScaler().fit(X_all[chosen])
            mu, sd = _predict_all(sc.transform(X_all[chosen]), Y_all[chosen],
                                  sc.transform(X_all[remaining]))
            base = np.array([_margin_of(mu[i]) for i in range(len(remaining))])
            if method == "bayes":
                unc = np.array([_margin_sd(sd[i]) for i in range(len(remaining))])
                acq = base + beta * unc
            else:
                acq = base
            pick = remaining[int(np.argmax(acq))]

        chosen.append(pick)
        history.append(float(true_score[chosen].max()))

    return np.array(history)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线选点实验")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data) if args.data else config.DATA_RAW
    out_dir = Path(args.out) if args.out else (config.ROOT / "outputs" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    if len(df) < INIT + 4:
        print(f"[失败] 数据只有 {len(df)} 组，至少要 {INIT + 4} 组", file=sys.stderr)
        return 1

    true_score = _scores(df)
    n = len(df)
    steps = min(args.steps, n - INIT)
    methods = {"random": "随机挑选", "grid": "网格扫描", "bayes": "贝叶斯（UCB）"}
    print(f"题库 {n} 组，初始 {INIT} 组，每轮加 1 组，共 {steps} 轮，重复 {args.seeds} 次\n")

    curves, finals = {}, {}
    for m in methods:
        runs = np.array([run_once(df, true_score, m, s, steps) for s in range(args.seeds)])
        curves[m] = runs.mean(0)
        finals[m] = list(runs[:, -1])
        print(f"  {methods[m]:<12} 最终最好余量 {runs[:, -1].mean():+.3f}，"
              f"≥{GOOD:.2f} 的比例 {np.mean(runs[:, -1] >= GOOD):.1%}")

    runs_mean = np.array([run_once(df, true_score, "bayes_mean", s, steps)
                          for s in range(args.seeds)])
    curves["bayes_mean"] = runs_mean.mean(0)
    finals["bayes_mean"] = list(runs_mean[:, -1])
    print(f"  {'贝叶斯（去不确定度）':<8} 最终最好余量 {runs_mean[:, -1].mean():+.3f}，"
          f"≥{GOOD:.2f} 的比例 {np.mean(runs_mean[:, -1] >= GOOD):.1%}")

    x = np.arange(len(curves["random"])) + INIT

    # ---------- 04 三组方法对比
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m, color in (("random", GREY), ("grid", ORANGE), ("bayes", BLUE)):
        ax.plot(x, curves[m], marker="o", ms=3.5, lw=1.8, color=color, label=methods[m])
    ax.axhline(GOOD, color=RED, lw=1.0, ls="--")
    ax.text(x[-1], GOOD + 0.01, f"余量 {GOOD:.2f}", ha="right", fontsize=8, color=RED)
    ax.set_xlabel("已仿真组数", fontsize=10)
    ax.set_ylabel("当前最好余量（越大越好）", fontsize=10)
    ax.set_title("三组方法：同样仿真次数下能达到的最好结果", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p04 = out_dir / "04_三组方法性能与成本对比.png"
    fig.savefig(p04, bbox_inches="tight")
    plt.close(fig)

    # ---------- 05 单次优化过程
    one = run_once(df, true_score, "bayes", 0, steps)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(one)) + INIT
    ax.plot(xs, one, marker="o", ms=5, lw=1.8, color=BLUE)
    ax.fill_between(xs, one.min() - 0.05, one, color=BLUE, alpha=0.08)
    ax.axhline(GOOD, color=RED, lw=1.0, ls="--")
    first_ok = next((i for i, v in enumerate(one) if v >= GOOD), None)
    if first_ok is not None:
        ax.annotate(f"第 {first_ok + INIT} 组首次达到 {GOOD:.2f}",
                    xy=(xs[first_ok], one[first_ok]),
                    xytext=(xs[first_ok] + 0.5, one[first_ok] - 0.15),
                    fontsize=8.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlabel("已仿真组数", fontsize=10)
    ax.set_ylabel("当前最好余量", fontsize=10)
    ax.set_title("一次完整的优化过程（种子 0）", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    p05 = out_dir / "05_优化过程曲线.png"
    fig.savefig(p05, bbox_inches="tight")
    plt.close(fig)

    # ---------- 06 达标率对照
    def rate_curve(m: str) -> np.ndarray:
        k = max(10, args.seeds // 2)
        runs = np.array([run_once(df, true_score, m, s, steps) for s in range(k)])
        return (runs >= GOOD).mean(0)

    r_bo, r_rand = rate_curve("bayes"), rate_curve("random")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, r_bo * 100, marker="o", ms=4, lw=1.8, color=BLUE, label="贝叶斯选点")
    ax.plot(x, r_rand * 100, marker="s", ms=4, lw=1.8, color=GREY, label="随机选点")
    ax.set_xlabel("已仿真组数", fontsize=10)
    ax.set_ylabel(f"余量达到 {GOOD:.2f} 的比例（%）", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("主动选点对照：多少组之后能找到好设计", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p06 = out_dir / "06_主动选点对照.png"
    fig.savefig(p06, bbox_inches="tight")
    plt.close(fig)

    # ---------- 07 消融
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, curves["bayes"], marker="o", ms=3.5, lw=1.8, color=BLUE,
            label="贝叶斯（含不确定度）")
    ax.plot(x, curves["bayes_mean"], marker="^", ms=3.5, lw=1.8, ls="--", color=ORANGE,
            label="去掉不确定度项")
    ax.axhline(GOOD, color=RED, lw=1.0, ls="--")
    ax.set_xlabel("已仿真组数", fontsize=10)
    ax.set_ylabel("当前最好余量", fontsize=10)
    if np.allclose(curves["bayes"], curves["bayes_mean"]):
        ax.text(0.5, 0.06, "两条曲线重合：当前数据量下，去掉不确定度项没有差别",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=RED)
    ax.set_title("消融实验：不确定度项到底有没有用", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p07 = out_dir / "07_误差反馈消融.png"
    fig.savefig(p07, bbox_inches="tight")
    plt.close(fig)

    # ---------- 08 重复运行统计
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    data = [finals["random"], finals["grid"], finals["bayes"]]
    bp = ax.boxplot(data, tick_labels=["随机", "网格扫描", "贝叶斯"],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], (GREY, ORANGE, BLUE)):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(GOOD, color=RED, lw=1.0, ls="--")
    ax.set_ylabel(f"{steps + INIT} 组后的最好余量", fontsize=10)
    ax.set_title(f"独立重复运行 {args.seeds} 次的结果分布", fontsize=11)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    fig.tight_layout()
    p08 = out_dir / "08_独立重复运行统计.png"
    fig.savefig(p08, bbox_inches="tight")
    plt.close(fig)

    for p in (p04, p05, p06, p07, p08):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
