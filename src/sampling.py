r"""
生成第一批参数组合（给 UltraEM 去跑）
======================================
做法：在三个变量的合法范围内，均匀撒 N 个点（拉丁超立方），
保证：数量准确、不越界、不重复、随机种子固定可复现。
为什么不用"网格"：三个变量凑 100 组，网格只能是 4×5×5，
每个变量只试了 4~5 个值，中间全空。撒点能让每个变量都试到 100 个不同值。
不训练模型、不预测、不判断设计好坏。
放哪里：B_Group/src/sampling.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
用 config 里的范围生成 100 组：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.sampling
指定数量和输出名：
    python -m src.sampling --n 40
范围还没填时，可以先用 --demo-ranges 试跑（只在屏幕上演示，不写文件）：
    python -m src.sampling --demo-ranges
被别的文件调用：
    from src.sampling import sample
    df = sample(n=100, seed=42)
常见报错：
    IncompleteRangeError: 还有 3 个变量没有范围
        → 仿真组还没给上下限，先用 --demo-ranges 看效果
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
try:
    from . import config
    from .search_space import IncompleteRangeError, get_ranges
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from search_space import IncompleteRangeError, get_ranges  # type: ignore
# ---------------------------------------------------------------- 撒点
def _latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """拉丁超立方：每维把 [0,1] 均分成 n 份，每份恰好取一个点，再打乱顺序。
    自己实现，不依赖 scipy，装不装 scipy 都能跑。
    """
    rng = np.random.default_rng(seed)
    # 每一维独立打乱"第几个区间"
    strata = np.argsort(rng.random((n, d)), axis=0)
    # 在每个区间内随机取一点
    jitter = rng.random((n, d))
    return (strata + jitter) / n
def sample(n: int = 100,
           seed: int = 42,
           ranges: dict[str, tuple[float, float]] | None = None
           ) -> pd.DataFrame:
    """在范围内撒 n 个点，返回带 candidate_id 的表。"""
    if n < 1:
        raise ValueError(f"n 必须 ≥ 1，收到 {n}")
    r = get_ranges(ranges)
    keys = list(r)
    unit = _latin_hypercube(n, len(keys), seed)   # n × d，值域 [0, 1)
    data: dict[str, object] = {"candidate_id": [f"{i+1:03d}" for i in range(n)]}
    for idx, key in enumerate(keys):
        lo, hi = r[key]
        data[key] = lo + unit[:, idx] * (hi - lo)
    df = pd.DataFrame(data)
    # 自检：不越界、不重复
    for key in keys:
        lo, hi = r[key]
        if df[key].min() < lo - 1e-9 or df[key].max() > hi + 1e-9:
            raise RuntimeError(f"{key} 有点越界了，检查范围设置")
    if df.drop(columns="candidate_id").duplicated().any():
        raise RuntimeError("出现了重复点，换一个 seed 或检查范围")
    return df
# ---------------------------------------------------------------- 命令行
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成第一批参数组合")
    parser.add_argument("--n", type=int, default=100, help="要几组，默认 100")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--out", type=str, default=None, help="输出文件名")
    parser.add_argument("--demo-ranges", action="store_true",
                        help="用示例范围演示，不写文件")
    args = parser.parse_args(argv)
    demo = {
        "c1_plate_len_um": (10.0, 40.0),
        "l1_r_in_um": (20.0, 80.0),
        "c2_plate_len_um": (10.0, 40.0),
    }
    try:
        ranges = demo if args.demo_ranges else None
        df = sample(n=args.n, seed=args.seed, ranges=ranges)
    except (IncompleteRangeError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        if not args.demo_ranges:
            print("提示：加 --demo-ranges 可以先看效果。", file=sys.stderr)
        return 1
    print(f"生成了 {len(df)} 组参数（seed={args.seed}）")
    for col in df.columns[1:]:
        print(f"  {col}: {df[col].min():.3f} ~ {df[col].max():.3f}")
    print()
    print(df.head(5).to_string(index=False))
    if args.demo_ranges:
        print("\n（演示模式：用的是示例范围，没有写文件）")
        return 0
    out = config.OUT_CANDIDATES / (args.out or f"initial_{args.n}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())