r"""
参数空间与边界检查
====================
只做三件事：
  1. 给出合法范围（从 config 读，读不到就报清楚缺什么）
  2. 判断一个点是否越界，越了哪一维
  3. 判断这个点离边界有多近（贴边的点要送真实仿真）
不生成点、不训练模型、不预测性能。
放哪里：B_Group/src/search_space.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
单独跑（用一个示例范围演示，不代表真实范围）：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.search_space
被别的文件调用：
    from src.search_space import get_ranges, check, check_many
    r = check({"c1_plate_len_um": 20.0, ...})      # 用 config 里的范围
    r = check(point, ranges={"c1_plate_len_um": (10, 30)})   # 临时指定
常见报错：
    IncompleteRangeError: 还有 3 个变量没有范围
        → 仿真组还没给上下限，正常，等他们
"""
from __future__ import annotations
from dataclasses import dataclass, field
try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore
class IncompleteRangeError(RuntimeError):
    """变量范围没填全时抛这个，提示缺哪一个。"""
# ---------------------------------------------------------------- 范围
def get_ranges(override: dict[str, tuple[float, float]] | None = None
               ) -> dict[str, tuple[float, float]]:
    """取每个变量的 (下限, 上限)。override 优先，方便测试和临时试算。"""
    if override is not None:
        return {k: (float(lo), float(hi)) for k, (lo, hi) in override.items()}
    missing = [v for v in config.VARIABLES if not v.is_ready]
    if missing:
        names = "\n".join(f"  - {v.name}（{v.key}）" for v in missing)
        raise IncompleteRangeError(
            f"还有 {len(missing)} 个变量没有范围，先找仿真组要：\n{names}"
        )
    return {v.key: (float(v.lower), float(v.upper)) for v in config.VARIABLES}
# ---------------------------------------------------------------- 判定
@dataclass(frozen=True)
class Feasibility:
    x: dict[str, float]
    ok: bool
    violations: tuple[str, ...] = ()        # 越界说明（空 = 没越界）
    nearest_frac: float = 0.5               # 离最近边界的位置比例：0=贴边，0.5=正中
    near_boundary: bool = False             # 是否贴边（建议送真实仿真）
    def reason(self) -> str:
        if not self.ok:
            return "越界：" + "；".join(self.violations)
        if self.near_boundary:
            return f"贴边（在 {self.nearest_frac:.1%} 处），建议真实仿真"
        return "范围与位置都正常"
def normalized(x: dict[str, float],
               ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """把物理值映射到 0~1。0 = 下界，1 = 上界。"""
    r = get_ranges(ranges)
    out: dict[str, float] = {}
    for key, (lo, hi) in r.items():
        if hi <= lo:
            raise ValueError(f"{key} 的范围上限不大于下限：({lo}, {hi})")
        out[key] = (float(x[key]) - lo) / (hi - lo)
    return out
def check(x: dict[str, float],
          ranges: dict[str, tuple[float, float]] | None = None,
          boundary_frac: float = 0.05) -> Feasibility:
    """检查一个点。boundary_frac=0.05 表示离边界 5% 以内算"贴边"。"""
    r = get_ranges(ranges)
    miss = [k for k in r if k not in x]
    if miss:
        raise KeyError(f"这个点缺参数：{miss}")
    violations: list[str] = []
    fracs: list[float] = []
    for key, (lo, hi) in r.items():
        val = float(x[key])
        if val < lo:
            violations.append(f"{key}={val:g} < 下限 {lo:g}")
        elif val > hi:
            violations.append(f"{key}={val:g} > 上限 {hi:g}")
        else:
            width = hi - lo
            pos = (val - lo) / width if width > 0 else 0.5
            fracs.append(min(pos, 1.0 - pos) * 2.0)   # 0=贴边, 1=正中（归一化到 0~1）
    ok = not violations
    nearest = min(fracs) / 2.0 if fracs else 0.0     # 换回 0~0.5 的"距边比例"
    near = ok and nearest < boundary_frac
    return Feasibility(
        x={k: float(x[k]) for k in r},
        ok=ok,
        violations=tuple(violations),
        nearest_frac=round(nearest, 4),
        near_boundary=near,
    )
def check_many(points: list[dict[str, float]],
               ranges: dict[str, tuple[float, float]] | None = None,
               boundary_frac: float = 0.05) -> list[Feasibility]:
    r = get_ranges(ranges)
    return [check(p, r, boundary_frac) for p in points]
def filter_ok(points: list[dict[str, float]],
              ranges: dict[str, tuple[float, float]] | None = None) -> list[dict[str, float]]:
    """只留下没越界的点，越界的直接扔（并告诉扔了几个）。"""
    results = check_many(points, ranges)
    kept = [f.x for f in results if f.ok]
    dropped = len(results) - len(kept)
    if dropped:
        print(f"[search_space] 丢掉越界点 {dropped} 个，保留 {len(kept)} 个")
    return kept
# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    print("提示：下面用的是示例范围，不是真实参数范围。\n")
    demo_ranges = {
        "c1_plate_len_um": (10.0, 40.0),
        "l1_r_in_um": (20.0, 80.0),
        "c2_plate_len_um": (10.0, 40.0),
    }
    samples = [
        {"c1_plate_len_um": 25.0, "l1_r_in_um": 50.0, "c2_plate_len_um": 25.0},  # 正常
        {"c1_plate_len_um": 25.0, "l1_r_in_um": 50.0, "c2_plate_len_um": 10.2},  # 贴边
        {"c1_plate_len_um": 55.0, "l1_r_in_um": 50.0, "c2_plate_len_um": 25.0},  # 越界
    ]
    for p in samples:
        f = check(p, demo_ranges)
        print(f"{p}  ->  {'OK' if f.ok else 'FAIL'}  {f.reason()}")