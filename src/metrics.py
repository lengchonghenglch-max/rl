r"""
指标计算：给一条 S 参数曲线，判断这个设计过没过
==================================================

输入：频率 + S11 / S21 / S22（复数或幅度都行）
输出：四条指标各自的最差值、是否达标、余量多少

这四条"最差值"同时就是 GP 要学的目标（4 个标量），所以这个文件
既用来判断合格，也用来生成训练标签。

放哪里：B_Group/src/metrics.py

------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
单独跑（打印一个手写示例，验证函数能算）：

    cd /d D:\rf\挑战杯\B_Group
    python -m src.metrics

被别的文件调用：

    from src.metrics import compute, Report
    report = compute(freq_hz, s11, s21, s22)
    print(report.passed, report.to_dict())

常见报错：

    ValueError: 频率数组和 S 参数长度不一致
        → 检查是不是把不同组的数据拼一起了
    ValueError: 频段 0.1–1.5 GHz 内没有频点
        → 频率范围或步长设错了
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # 支持 python -m src.metrics 和直接跑两种方式
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ---------------------------------------------------------------- 数据结构

@dataclass(frozen=True)
class MetricResult:
    key: str
    name: str
    value_db: float          # 该频段内最差的那个值
    limit_db: float
    passed: bool
    margin_db: float         # 正数 = 还有富余，负数 = 超了

    def line(self) -> str:
        op = "≤"
        return (
            f"{self.name:<8} {self.value_db:>8.2f} dB   {op} {self.limit_db:>6.1f}"
            f"   余量 {self.margin_db:>+7.2f} dB   {'PASS' if self.passed else 'FAIL'}"
        )


@dataclass
class Report:
    results: list[MetricResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def worst_margin_db(self) -> float:
        """所有指标里最紧的那条还剩多少余量。负 = 不合格。"""
        return min((r.margin_db for r in self.results), default=float("nan"))

    def to_dict(self) -> dict[str, float]:
        """给 GP 用的标量目标（4 个数）。"""
        return {r.key: r.value_db for r in self.results}

    def failures(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def format_table(self) -> str:
        head = f"{'指标':<8} {'数值':>10}      阈值        余量          结果"
        lines = [head, "-" * 72]
        lines += [r.line() for r in self.results]
        lines.append("-" * 72)
        lines.append(f"总判定：{'PASS' if self.passed else 'FAIL'}"
                     f"    最紧余量：{self.worst_margin_db:+.2f} dB")
        return "\n".join(lines)


# ---------------------------------------------------------------- 计算

def _db(mag: np.ndarray) -> np.ndarray:
    """20*log10(|x|)，顺手防一下 log(0) 变成 -inf。"""
    mag = np.maximum(np.abs(mag), 1e-12)
    return 20.0 * np.log10(mag)


def _band_mask(freq_hz: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freq_hz >= lo) & (freq_hz <= hi)


def compute(freq_hz, s11, s21, s22) -> Report:
    """四条指标一次算完。

    s11 / s21 / s22 可以是复数数组，也可以只给幅度；
    频率单位必须是 Hz。
    """
    freq = np.asarray(freq_hz, dtype=float)
    s11 = np.asarray(s11)
    s21 = np.asarray(s21)
    s22 = np.asarray(s22)

    if not (len(freq) == len(s11) == len(s21) == len(s22)):
        raise ValueError(
            f"频率数组和 S 参数长度不一致：{len(freq)}, {len(s11)}, {len(s21)}, {len(s22)}"
        )
    if len(freq) == 0:
        raise ValueError("输入是空的")

    # 每条 S 参数先转成 dB
    s11_db = _db(s11)
    s21_db = _db(s21)
    s22_db = _db(s22)

    # 指标 key → 用来判定的那条曲线
    curves: dict[str, np.ndarray] = {
        "il_db": -s21_db,      # 插损 = -S21(dB)，取正数
        "s11_db": s11_db,
        "s22_db": s22_db,
        "stop_db": s21_db,
    }

    results: list[MetricResult] = []
    for metric in config.METRICS:
        mask = _band_mask(freq, metric.band)
        if not mask.any():
            lo, hi = metric.band
            raise ValueError(
                f"频段 {lo/1e9:g}–{hi/1e9:g} GHz 内没有频点，检查频率网格"
            )

        arr = curves[metric.key][mask]
        worst = float(arr.max() if metric.direction == "max" else arr.min())

        if metric.direction == "max":
            passed = worst <= metric.limit_db
            margin = metric.limit_db - worst
        else:
            passed = worst >= metric.limit_db
            margin = worst - metric.limit_db

        results.append(
            MetricResult(
                key=metric.key,
                name=metric.name,
                value_db=round(worst, 4),
                limit_db=metric.limit_db,
                passed=passed,
                margin_db=round(margin, 4),
            )
        )

    return Report(results=results)


# ---------------------------------------------------------------- 自检

if __name__ == "__main__":
    print("提示：这是一条手写的示例曲线，只用来验证函数能算，不是训练数据。\n")

    freq = config.FREQ_GRID
    n = len(freq)

    # 故意做成：通带合格、阻带不合格
    s21 = np.full(n, 0.85 + 0j)   # -1.41 dB
    s11 = np.full(n, 0.20 + 0j)   # -14.0 dB
    s22 = np.full(n, 0.20 + 0j)

    report = compute(freq, s11, s21, s22)
    print(report.format_table())
    print()
    print("给 GP 用的 4 个标量：", report.to_dict())
