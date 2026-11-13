r"""
安全检查：哪些候选点必须回 UltraEM
==================================
三件事：
    1. 硬边界：超出已验证范围 → 直接禁止
    2. 贴边：离边界太近 → 必须仿真
    3. 没把握：模型不确定度大，或附近没有已仿真的点 → 必须仿真
说明：现在的"已验证范围"取自已有数据的 min/max；等配置里的参数域确定后，
以配置为准（config.VARIABLES 填了就用它）。
放哪里：B_Group/src/safety_check.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
先看一份推荐表安不安全：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.safety_check --input outputs\recommendations推荐参数\next_batch.csv
常见报错：
    [失败] 推荐表缺列
        → 表里要有 c1_pf、c2_pf 两列
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import config
    from .gp_model import scan_dataset
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from gp_model import scan_dataset  # type: ignore
# 判定阈值（先给经验值，等真实数据多了再校准）
BOUNDARY_FRAC = 0.05      # 离边界 5% 以内算贴边
COVERAGE_FRAC = 0.10      # 离最近已知点超过范围的 10% 算数据稀疏
UNCERTAINTY_SD = 0.15     # 归一化余量不确定度超过这个值算没把握
BAN = "禁止"
MUST = "必须仿真"
OK = "可参考"
def _ranges(df_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        "c1_pf": (float(df_train.c1_pf.min()), float(df_train.c1_pf.max())),
        "c2_pf": (float(df_train.c2_pf.min()), float(df_train.c2_pf.max())),
    }
def check_point(c1: float, c2: float, ranges: dict, train: np.ndarray,
                margin_sd: float | None = None) -> tuple[str, str]:
    reasons = []
    # 1. 硬边界
    for name, val in (("c1_pf", c1), ("c2_pf", c2)):
        lo, hi = ranges[name]
        if val < lo - 1e-9 or val > hi + 1e-9:
            reasons.append(f"{name}={val:g} 超出已验证范围 [{lo:g}, {hi:g}]")
    if reasons:
        return BAN, "；".join(reasons)
    # 2. 贴边
    near = []
    for name, val in (("c1_pf", c1), ("c2_pf", c2)):
        lo, hi = ranges[name]
        width = hi - lo if hi > lo else 1.0
        pos = (val - lo) / width
        if min(pos, 1 - pos) < BOUNDARY_FRAC:
            near.append(f"{name} 贴近边界（{pos:.1%} 处）")
    # 3. 数据覆盖
    span = max(ranges["c1_pf"][1] - ranges["c1_pf"][0],
               ranges["c2_pf"][1] - ranges["c2_pf"][0], 1e-9)
    d = float(np.min(np.linalg.norm(train - np.array([c1, c2]), axis=1)))
    sparse = d / span > COVERAGE_FRAC
    # 4. 模型不确定度
    unsure = margin_sd is not None and margin_sd > UNCERTAINTY_SD
    if near:
        reasons += near
    if sparse:
        reasons.append(f"附近无已知点（最近 {d:.2f} pF）")
    if unsure:
        reasons.append(f"预测不确定度偏高（{margin_sd:.2f}）")
    return (MUST if reasons else OK), ("；".join(reasons) if reasons else "范围内，位置正常")
def check_frame(df: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    ranges = _ranges(df_train)
    train = df_train[["c1_pf", "c2_pf"]].to_numpy(float)
    rows = []
    for _, r in df.iterrows():
        verdict, reason = check_point(
            float(r["c1_pf"]), float(r["c2_pf"]), ranges, train,
            float(r["margin_sd"]) if "margin_sd" in r and pd.notna(r.get("margin_sd")) else None,
        )
        rows.append({"c1_pf": r["c1_pf"], "c2_pf": r["c2_pf"], "结论": verdict, "理由": reason})
    return pd.DataFrame(rows)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="候选点安全检查")
    parser.add_argument("--input", type=str, required=True, help="推荐表 csv")
    parser.add_argument("--data", type=str, default=None, help="已知数据目录")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    if not {"c1_pf", "c2_pf"} <= set(df.columns):
        print("[失败] 推荐表缺列：需要有 c1_pf、c2_pf", file=sys.stderr)
        return 1
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    try:
        df_train = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    out = check_frame(df, df_train)
    print(out.to_string(index=False))
    print()
    for verdict in (BAN, MUST, OK):
        n = int((out["结论"] == verdict).sum())
        print(f"  {verdict}：{n} 组")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

