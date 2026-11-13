r"""
选点：从候选里挑出下一批最值得做仿真的组合
==========================================
先说清楚目标函数，这是选点的前提：
    每一项指标都有一个阈值，把"离阈值还有多少余量"归一化，
    取四项里最紧的那一项作为这组参数的成绩（margin）。
    margin > 0 表示预测合格，越大越安全。
选点就是在这个成绩上做 UCB：
    分数 = 预测余量 + beta × 余量的不确定度
beta 小 → 偏 exploitation（挑预测最好的）；beta 大 → 偏 exploration（挑最不确定的）。
再用 candidate 之间的最小距离做去重，避免一批点挤在一起。
放哪里：B_Group/src/acquisition.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
用已训练的模型生成下一批 5 个候选：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.acquisition --n 5
指定采样数、beta、去重距离：
    python -m src.acquisition --n 8 --beta 1.0 --min-dist 0.03
常见报错：
    [失败] 模型文件不存在
        → 先跑 python -m src.gp_model --save
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import config
    from .gp_model import TARGETS, GPModel, scan_dataset
    from .predict import margin
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from gp_model import TARGETS, GPModel, scan_dataset  # type: ignore
    from predict import margin  # type: ignore
def _limits() -> dict[str, float]:
    return {m.key: m.limit_db for m in config.METRICS}
def score_of(values: dict[str, float]) -> float:
    return margin(values)
def score_std(sds: dict[str, float]) -> float:
    """余量的不确定度：取四项里最紧那项的不确定度（按阈值归一化）。"""
    lim = _limits()
    return float(min(sds[k] / abs(lim[k]) for k in TARGETS))
def candidate_grid(lo: float, hi: float, n: int) -> np.ndarray:
    v = np.linspace(lo, hi, n)
    return np.array([[a, b] for a in v for b in v])
def rank(model: GPModel, df: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """给一批候选打分：预测余量、余量不确定度、UCB 分数。"""
    rows = []
    for _, r in df.iterrows():
        out = model.predict(float(r["c1_pf"]), float(r["c2_pf"])).iloc[0]
        vals = {k: float(out[k]) for k in TARGETS}
        sds = {k: float(out[f"{k}_sd"]) for k in TARGETS}
        m = score_of(vals)
        s = score_std(sds)
        rows.append({
            "c1_pf": float(r["c1_pf"]),
            "c2_pf": float(r["c2_pf"]),
            **vals,
            "margin": m,
            "margin_sd": s,
            "ucb": m + beta * s,
        })
    return pd.DataFrame(rows)
def pick(df_ranked: pd.DataFrame, n: int, min_dist: float) -> pd.DataFrame:
    """按 ucb 从高到低选，且与已选点保持最小距离。"""
    pool = df_ranked.sort_values("ucb", ascending=False).reset_index(drop=True)
    chosen = []
    for _, r in pool.iterrows():
        if len(chosen) >= n:
            break
        p = np.array([r["c1_pf"], r["c2_pf"]])
        if any(np.linalg.norm(p - np.array([c["c1_pf"], c["c2_pf"]])) < min_dist for c in chosen):
            continue
        chosen.append(r)
    if not chosen:
        return pool.head(n).copy()
    out = pd.DataFrame(chosen).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="选下一批候选参数")
    parser.add_argument("--n", type=int, default=5, help="要几组，默认 5")
    parser.add_argument("--beta", type=float, default=0.5, help="UCB 的探索权重")
    parser.add_argument("--min-dist", type=float, default=0.05, help="批内最小间距（pF）")
    parser.add_argument("--grid", type=int, default=40, help="候选网格密度")
    parser.add_argument("--data", type=str, default=None, help="已知数据目录")
    parser.add_argument("--save", action="store_true", help="保存推荐表")
    args = parser.parse_args(argv)
    model_path = config.MODEL_DIR / "gp_model.joblib"
    if not model_path.exists():
        print(f"[失败] 模型文件不存在：{model_path}", file=sys.stderr)
        print("先跑：python -m src.gp_model --save", file=sys.stderr)
        return 1
    model = GPModel.load(model_path)
    # 候选范围：有配置用配置，没有就用已知数据的范围
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    df_train = scan_dataset(data_dir)
    lo1, hi1 = df_train.c1_pf.min(), df_train.c1_pf.max()
    lo2, hi2 = df_train.c2_pf.min(), df_train.c2_pf.max()
    grid = candidate_grid(lo1, hi1, args.grid) if lo1 == lo2 and hi1 == hi2 else np.array(
        [[a, b] for a in np.linspace(lo1, hi1, args.grid) for b in np.linspace(lo2, hi2, args.grid)])
    cand = pd.DataFrame(grid, columns=["c1_pf", "c2_pf"])
    ranked = rank(model, cand, beta=args.beta)
    picked = pick(ranked, args.n, args.min_dist)
    print(f"候选 {len(cand)} 组，选 {len(picked)} 组（beta={args.beta}，最小间距 {args.min_dist} pF）")
    print(picked.drop(columns=["margin_sd"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    if args.save:
        out = config.OUT_RECOMMENDATIONS / "next_batch.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        picked.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已保存：{out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

