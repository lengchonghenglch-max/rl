r"""
B 组配置文件：全部"已知条件"都只在这里写一次
==============================================
这个文件里只有**常量**，没有任何计算。算指标的在 metrics.py，算范围的
在 range_estimate.py，跑全流程的在 main.py。
改参数、改指标，只改这个文件。
放哪里：B_Group/src/config.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
（这个文件一般不用单独跑，但可以跑一下看看配置有没有写对）
    cd /d D:\rf\挑战杯\B_Group
    python -m src.config
会打印：频率网格、三组指标、三个变量、还缺哪些信息。
被别的文件调用：
    from src.config import FREQ_GRID, METRICS, VARIABLES, missing_items
常见报错：
    ModuleNotFoundError: No module named 'numpy'
        → 没装依赖，先 pip install numpy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
# ================================================================
# 1. 频率设置（队长已定：0.1 ~ 8 GHz，步长 ≤50 MHz）
# ================================================================
F_START_HZ: float = 0.100e9
F_STOP_HZ: float = 8.000e9
F_STEP_HZ: float = 0.050e9
# 这几个频点必须出现在网格里（指标要按它们分段）
ANCHOR_FREQS_HZ: tuple[float, ...] = (0.1e9, 1.5e9, 6.0e9, 8.0e9)
def _build_freq_grid() -> np.ndarray:
    """0.1 → 8 GHz，步长 50 MHz，共 159 点。端点强制包含。"""
    n = int(round((F_STOP_HZ - F_START_HZ) / F_STEP_HZ)) + 1
    grid = np.linspace(F_START_HZ, F_STOP_HZ, n)
    for anchor in ANCHOR_FREQS_HZ:
        if not np.any(np.isclose(grid, anchor, atol=1.0)):
            raise ValueError(f"频点 {anchor/1e9:g} GHz 不在网格里，检查步长设置")
    return grid
FREQ_GRID: np.ndarray = _build_freq_grid()
N_FREQ: int = len(FREQ_GRID)          # 应该是 159
# ================================================================
# 2. 频段划分（队长已定）
# ================================================================
PASSBAND_HZ: tuple[float, float] = (0.1e9, 1.5e9)
STOPBAND_HZ: tuple[float, float] = (6.0e9, 8.0e9)
TRANSITION_HZ: tuple[float, float] = (1.5e9, 6.0e9)   # 首轮不设硬性约束
# ================================================================
# 3. 指标（队长已定，单位 dB）
# ================================================================
Direction = Literal["max", "min"]   # max: 不超过 limit；min: 不低于 limit
@dataclass(frozen=True)
class Metric:
    """一条硬性指标。band 是判定用的频段，param 是从哪条 S 参数算。"""
    key: str
    name: str
    param: Literal["s11", "s21", "s22"]
    band: tuple[float, float]
    direction: Direction
    limit_db: float
    @property
    def band_name(self) -> str:
        lo, hi = self.band
        return f"{lo/1e9:g}–{hi/1e9:g} GHz"
METRICS: tuple[Metric, ...] = (
    Metric("il_db",   "通带插入损耗", "s21", PASSBAND_HZ, "max", 3.0),
    Metric("s11_db",  "通带输入回波", "s11", PASSBAND_HZ, "max", -10.0),
    Metric("s22_db",  "通带输出回波", "s22", PASSBAND_HZ, "max", -10.0),
    Metric("stop_db", "阻带衰减",     "s21", STOPBAND_HZ, "max", -12.0),
)
# ================================================================
# 4. 优化变量（电磁仿真 → 变量是几何尺寸，不是 pF / nH）
# ================================================================
@dataclass(frozen=True)
class Variable:
    key: str
    name: str
    unit: str
    lower: float | None      # None = 还没从仿真组拿到
    upper: float | None
    note: str = ""
    @property
    def is_ready(self) -> bool:
        return self.lower is not None and self.upper is not None
VARIABLES: tuple[Variable, ...] = (
    Variable("c1_plate_len_um", "C1 电容极板长度", "um", None, None,
             "TODO 等仿真组给上下限"),
    Variable("l1_r_in_um",      "L1 电感内半径",   "um", None, None,
             "TODO 等仿真组给上下限"),
    Variable("c2_plate_len_um", "C2 电容极板长度", "um", None, None,
             "TODO 等仿真组给上下限（与 C1 同范围吗？）"),
)
VARIABLE_KEYS: tuple[str, ...] = tuple(v.key for v in VARIABLES)
N_VARIABLES: int = len(VARIABLES)     # 3
# 理想电路值：只用来核对，不作为模型输入
REFERENCE_ELECTRICAL: dict[str, tuple[float, str]] = {
    "c1_pf": (1.06, "pF"),
    "l1_nh": (5.31, "nH"),
    "c2_pf": (1.06, "pF"),
}
# ================================================================
# 5. 还没拿到的信息（main.py 会打印出来当前卡在哪）
# ================================================================
STACKUP: dict | None = None    # 层叠：材料、εr、各层厚度、金属厚度 —— TODO 等仿真组
DATA_SOURCE_FORMAT: str | None = None   # "touchstone" / "csv" / None（未知）
REQUIRED_FROM_SIM_TEAM: tuple[str, ...] = (
    "三个几何变量的上下限（μm）",
    "层叠参数（材料、εr、各层厚度、金属厚度）",
    "一份真实导出文件（确认格式、单位、复数表示）",
    "单次仿真耗时 / 能否批量 / 失败时输出什么",
)
# ================================================================
# 6. 数据标签（三类数据不能混，贯穿数据库、页面、报告）
# ================================================================
DataLabel = Literal["simulation", "prediction", "validation"]
DATA_LABELS: tuple[DataLabel, ...] = ("simulation", "prediction", "validation")
# ================================================================
# 7. 目录（跟着你实际的中文文件夹名走）
# ================================================================
from pathlib import Path
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw原数据"
DATA_PROCESSED: Path = ROOT / "data" / "processed整理后数据"
DATA_DEMO: Path = ROOT / "data" / "demo测试用数据集"
OUT_CANDIDATES: Path = ROOT / "outputs" / "candidates候选参数"
OUT_PREDICTIONS: Path = ROOT / "outputs" / "predictions预测结果"
OUT_RECOMMENDATIONS: Path = ROOT / "outputs" / "recommendations推荐参数"
OUT_VALIDATION: Path = ROOT / "outputs" / "validation仿真验证结果"
MODEL_DIR: Path = ROOT / "models"
# ================================================================
# 8. 检查函数：现在能不能开工
# ================================================================
def missing_items() -> list[str]:
    """返回还缺什么。空列表 = 可以正式跑真实数据了。"""
    missing: list[str] = []
    for v in VARIABLES:
        if not v.is_ready:
            missing.append(f"{v.name}（{v.key}）范围未填：{v.note}")
    if STACKUP is None:
        missing.append("层叠参数未填（STACKUP）")
    if DATA_SOURCE_FORMAT is None:
        missing.append("仿真导出格式未确认（DATA_SOURCE_FORMAT）")
    return missing
def summary(show_missing: bool = True) -> str:
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("B 组配置")
    lines.append("=" * 62)
    lines.append(
        f"频率网格：{F_START_HZ/1e9:g} – {F_STOP_HZ/1e9:g} GHz，"
        f"步长 {F_STEP_HZ/1e6:g} MHz，共 {N_FREQ} 点"
    )
    lines.append(f"通带：{PASSBAND_HZ[0]/1e9:g} – {PASSBAND_HZ[1]/1e9:g} GHz")
    lines.append(f"阻带：{STOPBAND_HZ[0]/1e9:g} – {STOPBAND_HZ[1]/1e9:g} GHz")
    lines.append(f"过渡带：{TRANSITION_HZ[0]/1e9:g} – {TRANSITION_HZ[1]/1e9:g} GHz（无硬约束）")
    lines.append("")
    lines.append("指标：")
    for m in METRICS:
        op = "≤" if m.direction == "max" else "≥"
        lines.append(
            f"  - {m.name}（{m.param.upper()}，{m.band_name}）：{op} {m.limit_db:g} dB"
        )
    lines.append("")
    lines.append(f"优化变量（{N_VARIABLES} 个，单位 μm）：")
    for v in VARIABLES:
        if v.is_ready:
            rng = f"[{v.lower:g}, {v.upper:g}]"
        else:
            rng = "[未填]"
        lines.append(f"  - {v.name}（{v.key}）：{rng} {v.unit}")
    lines.append("")
    lines.append("理论电路值（仅核对，不是模型输入）：")
    for k, (val, unit) in REFERENCE_ELECTRICAL.items():
        lines.append(f"  - {k}: {val:g} {unit}")
    lines.append("")
    miss = missing_items()
    if show_missing:
        if miss:
            lines.append(f"还缺 {len(miss)} 项，暂时只能跑通不依赖数据的部分：")
            for m in miss:
                lines.append(f"  ✗ {m}")
        else:
            lines.append("配置齐全，可以跑真实数据。")
    lines.append("=" * 62)
    return "\n".join(lines)
if __name__ == "__main__":
    print(summary())
