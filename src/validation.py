r"""
独立验证：预测值和真实仿真对不对得上
====================================
做两件事：
    1. 把"预测表"和"新仿真数据"按参数配对
    2. 算误差，看预测的合格/不合格判断有没有和应用一致
注意：这里的"真实数据"必须是独立重跑得到的，不能拿训练集里的点充数。
放哪里：B_Group/src/validation.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
    cd /d D:\rf\挑战杯\B_Group
    python -m src.validation --pred outputs\predictions预测结果\next_batch_pred.csv ^
                              --data data\raw原数据
常见报错：
    [失败] 预测表和仿真数据没有匹配上
        → 两边的 c1/c2 数值必须一致（比如 0.3 和 0.30 算同一个，但 0.3 和 0.35 不是）
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import config
    from .gp_model import TARGETS, scan_dataset
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from gp_model import TARGETS, scan_dataset  # type: ignore
def _key(df: pd.DataFrame) -> pd.Series:
    return (df["c1_pf"].round(4).astype(str) + "|" + df["c2_pf"].round(4).astype(str))
def compare(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    truth = truth.copy()
    pred["_k"] = _key(pred)
    truth["_k"] = _key(truth)
    merged = pred.merge(truth, on="_k", suffixes=("_pred", "_true"))
    if merged.empty:
        raise ValueError("预测表和仿真数据没有匹配上：两边的 c1/c2 没有相同的组合")
    rows = []
    for _, r in merged.iterrows():
        row = {"c1_pf": r["c1_pf_pred"], "c2_pf": r["c2_pf_pred"]}
        for t in TARGETS:
            p, a = float(r[f"{t}_pred"]), float(r[f"{t}_true"])
            row[f"{t}_pred"] = p
            row[f"{t}_true"] = a
            row[f"{t}_err"] = p - a
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"匹配上 {len(out)} 组")
    print()
    print("误差统计（预测 − 真实）")
    stat = []
    for t in TARGETS:
        e = out[f"{t}_err"].to_numpy(float)
        stat.append({"指标": t, "平均误差": e.mean(), "MAE": np.abs(e).mean(),
                     "RMSE": np.sqrt((e ** 2).mean()), "最大误差": np.abs(e).max()})
    print(pd.DataFrame(stat).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预测 vs 独立仿真")
    parser.add_argument("--pred", type=str, required=True, help="预测表 csv")
    parser.add_argument("--data", type=str, default=None, help="新仿真数据目录")
    parser.add_argument("--save", action="store_true", help="保存对比明细")
    args = parser.parse_args(argv)
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    try:
        truth = scan_dataset(data_dir)
        pred = pd.read_csv(args.pred)
        out = compare(pred, truth)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    if args.save:
        path = config.OUT_VALIDATION / "compare.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n已保存：{path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
