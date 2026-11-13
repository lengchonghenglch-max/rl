r"""
出图：频响曲线（横轴频率、纵轴 dB）
==================================
生成申报材料的前三张图：
    01_S参数曲线.png                     0.1–8 GHz 全频段，S11 / S21 / S22
    02_通带损耗与输入输出回波损耗.png      通带 0.1–1.5 GHz，上下两栏：插损 / 回波
    03_阻带衰减.png                       阻带 6–8 GHz，合格与不合格设计对比
放哪里：B_Group/src/visualize.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
    cd /d D:\rf\挑战杯\B_Group
    python -m src.visualize
只出一张：
    python -m src.visualize --only 02
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
from matplotlib.ticker import FuncFormatter
try:
    from . import config
    from .data_process import read_touchstone
    from .gp_model import scan_dataset
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from data_process import read_touchstone  # type: ignore
    from gp_model import scan_dataset  # type: ignore
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
GREY = "#9aa5ad"
def _db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))
def _curves(row: pd.Series):
    ts = read_touchstone(row["source"])
    f = ts.freq_hz / 1e9
    return f, _db(ts.s[(1, 1)]), _db(ts.s[(2, 1)]), _db(ts.s[(2, 2)])
def _best_and_worst(df: pd.DataFrame):
    ok = df[df.passed].copy()
    bad = df[~df.passed].copy()
    best = ok.sort_values("stop_db").iloc[0]
    worst = bad.sort_values("stop_db", ascending=False).iloc[0]
    return best, worst
def fig_s_full(df: pd.DataFrame, out: Path) -> Path:
    # 选最靠近数据中心的合格设计：四个角上的点属外推，画出来参考价值不大
    center = df[df.passed].copy()
    center["dist"] = (center.c1_pf - df.c1_pf.median()).abs() + (center.c2_pf - df.c2_pf.median()).abs()
    best = center.sort_values("dist").iloc[0]
    # 留一法：把这一组从训练集里去掉再预测，才是诚实的"预测曲线"
    try:
        from .curve_model import CurveModel
    except ImportError:  # pragma: no cover
        from curve_model import CurveModel  # type: ignore
    cm = CurveModel().fit(df.drop(index=best.name))
    pred = cm.predict(float(best.c1_pf), float(best.c2_pf))
    f_pred = pred["s21"]["freq_hz"] / 1e9
    f, s11, s21, s22 = _curves(best)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(0.1, 1.5, color="#e8f1f3", alpha=0.9, zorder=0)
    ax.axvspan(6.0, 8.0, color="#f7ecec", alpha=0.9, zorder=0)
    lo = pred["s21"]["mean_db"] - pred["s21"]["sd_db"]
    hi = pred["s21"]["mean_db"] + pred["s21"]["sd_db"]
    ax.fill_between(f_pred, lo, hi, color=BLUE, alpha=0.20, lw=0, zorder=1)
    ax.plot(f, s21, color=BLUE, lw=2.0, label="S21 实测", zorder=3)
    ax.plot(f_pred, pred["s21"]["mean_db"], color=BLUE, lw=1.2, ls="--",
            label="S21 预测", zorder=3)
    ax.plot(f, s11, color=ORANGE, lw=1.3, label="S11 实测", zorder=2)
    ax.plot(f_pred, pred["s11"]["mean_db"], color=ORANGE, lw=1.0, ls="--",
            label="S11 预测", zorder=2)
    ax.plot(f, s22, color=GREEN, lw=1.3, ls="-.", label="S22 实测", zorder=2)
    ax.plot(f_pred, pred["s22"]["mean_db"], color=GREEN, lw=1.0, ls=":",
            label="S22 预测", zorder=2)
    for y, txt in ((-3, "−3 dB"), (-10, "−10 dB")):
        ax.axhline(y, color=GREY, lw=0.8, ls=":", zorder=1)
        ax.text(0.15, y + 0.8, txt, fontsize=7.5, color=GREY, va="bottom")
    ax.text(0.8, 6.5, "通带 0.1–1.5 GHz", ha="center", fontsize=9, color=BLUE)
    ax.text(7.0, 6.5, "阻带 6–8 GHz", ha="center", fontsize=9, color=RED)
    ax.set_xlim(0.1, 8)
    ax.set_ylim(-45, 10)
    ax.set_xticks([0.1, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("频率 (GHz)", fontsize=10)
    ax.set_ylabel("幅值 (dB)", fontsize=10)
    ax.set_title(
        f"S 参数曲线（C1 = {best.c1_pf:g} pF，C2 = {best.c2_pf:g} pF；"
        f"实线实测，虚线留一法预测）", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    path = out / "01_S参数曲线.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
def fig_s_passband(df: pd.DataFrame, out: Path) -> Path:
    best, _ = _best_and_worst(df)
    f, s11, s21, s22 = _curves(best)
    m = (f >= 0.1) & (f <= 1.5)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.12})
    # 上：插损（合格线在 -3 dB，曲线贴在 0 附近）
    ax1.plot(f[m], s21[m], color=BLUE, lw=2.2)
    ax1.axhline(-3, color=RED, lw=1.0, ls="--")
    ax1.set_ylim(-3.4, 0.35)
    ax1.set_ylabel("S21 (dB)", fontsize=9.5)
    ax1.text(0.16, -2.9, "上限 −3 dB", fontsize=8, color=RED, va="bottom")
    ax1.text(1.46, -0.75, f"最差插损 {best.il_db:.2f} dB", fontsize=9, color=BLUE,
             ha="right", va="top")
    ax1.set_title(f"通带 0.1–1.5 GHz（C1 = {best.c1_pf:g} pF，C2 = {best.c2_pf:g} pF）",
                  fontsize=11)
    ax1.grid(alpha=0.25, lw=0.6)
    # 下：回波
    ax2.plot(f[m], s11[m], color=ORANGE, lw=1.6, label="S11（输入回波）")
    ax2.plot(f[m], s22[m], color=GREEN, lw=1.6, ls="--", label="S22（输出回波）")
    ax2.axhline(-10, color=RED, lw=1.0, ls="--")
    ax2.set_ylim(-30, -5)
    ax2.set_ylabel("S11 / S22 (dB)", fontsize=9.5)
    ax2.set_xlabel("频率 (GHz)", fontsize=10)
    ax2.text(0.16, -9.7, "上限 −10 dB", fontsize=8, color=RED, va="bottom")
    ax2.text(1.46, -14.5, f"最差回波 {max(best.s11_db, best.s22_db):.1f} dB",
             fontsize=9, color=ORANGE, ha="right", va="bottom")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax2.set_xlim(0.1, 1.5)
    fig.tight_layout()
    path = out / "02_通带损耗与输入输出回波损耗.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
def fig_s_stopband(df: pd.DataFrame, out: Path) -> Path:
    ok = df[df.passed].sort_values("stop_db")
    bad = df[~df.passed].sort_values("stop_db", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhspan(-12, 5, color="#f9e9e9", alpha=0.8, zorder=0)
    # 背景：全部曲线，淡淡一层
    for _, row in df.iterrows():
        f, _, s21, _ = _curves(row)
        m = (f >= 6.0) & (f <= 8.0)
        ax.plot(f[m], s21[m], color=GREY, lw=0.5, alpha=0.25, zorder=1)
    for i, (_, row) in enumerate(ok.head(3).iterrows()):
        f, _, s21, _ = _curves(row)
        m = (f >= 6.0) & (f <= 8.0)
        ax.plot(f[m], s21[m], color=GREEN, lw=1.8, zorder=3,
                label="合格设计（3 例）" if i == 0 else None)
    for i, (_, row) in enumerate(bad.head(3).iterrows()):
        f, _, s21, _ = _curves(row)
        m = (f >= 6.0) & (f <= 8.0)
        ax.plot(f[m], s21[m], color=RED, lw=1.8, ls="--", zorder=3,
                label="不合格设计（3 例）" if i == 0 else None)
    ax.axhline(-12, color=RED, lw=1.2, ls="-.")
    ax.text(6.05, -12.6, "达标线 −12 dB（线以下为达标）", fontsize=8, color=RED, va="top")
    ax.text(6.05, -3.6, "未达标区域", fontsize=8.5, color=RED, alpha=0.85)
    ax.set_xlim(6.0, 8.0)
    ax.set_ylim(-33, -2)
    ax.set_xlabel("频率 (GHz)", fontsize=10)
    ax.set_ylabel("S21 (dB)", fontsize=10)
    ax.set_title("阻带 6–8 GHz：合格与不合格设计对比", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    path = out / "03_阻带衰减.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成频响曲线图")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--out", type=str, default=None, help="输出目录")
    parser.add_argument("--only", choices=["01", "02", "03"], help="只出一张")
    args = parser.parse_args(argv)
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    out_dir = Path(args.out) if args.out else (config.ROOT / "outputs" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    jobs = {"01": fig_s_full, "02": fig_s_passband, "03": fig_s_stopband}
    for key, fn in jobs.items():
        if args.only and key != args.only:
            continue
        print(f"  {fn(df, out_dir)}")
    print(f"\n输出目录：{out_dir}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
