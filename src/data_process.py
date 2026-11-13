r"""
读 UltraEM 导出的 Touchstone 文件（.s1p / .s2p / .s4p …）
==========================================================
只做三件事：
  1. 把文件读成"频率 + 复数 S 矩阵"
  2. 检查数据干不干净（频率递增、缺列、NaN、频点不齐）
  3. 转成一张长表，方便后面训练和使用
不训练模型、不改原始文件。
放哪里：B_Group/src/data_process.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
读一个文件，打印检查结果：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.data_process --file "D:\某个路径\Design1.s2p"
读一个文件并转成长表（输出到 data\processed整理后数据\）：
    python -m src.data_process --file "...\Design1.s2p" --candidate-id 001 --save
不给 --file 时，会自动找 data\raw原数据\ 下的第一个 .s2p。
被别的文件调用：
    from src.data_process import read_touchstone, to_long_table
    ts = read_touchstone(path)
    df = to_long_table(ts, candidate_id="001", params={"c1_plate_len_um": 25.0})
常见报错：
    ValueError: 只支持 S 参数文件，这个文件是 Y 参数
        → 导出时选 S 参数
    ValueError: 第 5 行的列数和第一行不一致
        → 文件被截断了，或混了两个端口数的数据
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore
# ---------------------------------------------------------------- 数据结构
@dataclass
class Touchstone:
    path: Path
    freq_hz: np.ndarray
    s: dict[tuple[int, int], np.ndarray]   # (i, j) -> 复数数组
    z0: float = 50.0
    fmt: str = "RI"                       # RI / MA / DB
    n_ports: int = 2
    def matrix_names(self) -> list[str]:
        return [f"s{i}{j}" for (i, j) in sorted(self.s)]
@dataclass
class CheckResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    def format(self) -> str:
        lines = []
        for n in self.notes:
            lines.append(f"  ✓ {n}")
        for i in self.issues:
            lines.append(f"  ✗ {i}")
        return "\n".join(lines) if lines else "  （没有检查项）"
# ---------------------------------------------------------------- 解析
_FREQ_SCALE = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
def read_touchstone(path: str | Path) -> Touchstone:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    freq_scale = 1e9      # Touchstone 默认 GHz
    fmt = "MA"            # 默认 MA
    ptype = "S"           # 默认 S
    z0 = 50.0
    rows: list[list[float]] = []
    ncol: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                tokens = line[1:].upper().split()
                for t in tokens:
                    if t in _FREQ_SCALE:
                        freq_scale = _FREQ_SCALE[t]
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t in ("S", "Y", "Z", "H", "G"):
                        ptype = t
                    elif t == "R":
                        continue
                # R 后面的数值单独处理
                parts = line[1:].split()
                for i, p in enumerate(parts[:-1]):
                    if p.upper() == "R":
                        try:
                            z0 = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                continue
            try:
                vals = [float(x) for x in line.replace(",", " ").split()]
            except ValueError:
                raise ValueError(f"第 {lineno} 行有非数字内容：{line[:60]}")
            if ncol is None:
                ncol = len(vals)
                if (ncol - 1) % 2 != 0:
                    raise ValueError(
                        f"第 {lineno} 行有 {ncol} 列，应该是 1 + 2×N²（频率 + 实部/虚部）"
                    )
            elif len(vals) != ncol:
                raise ValueError(
                    f"第 {lineno} 行的列数和第一行不一致（{len(vals)} vs {ncol}）。"
                    f"文件可能被截断，或混了不同端口数的数据。"
                )
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path.name} 里没有任何数据行")
    if ptype != "S":
        raise ValueError(f"只支持 S 参数文件，这个文件是 {ptype} 参数")
    arr = np.asarray(rows, dtype=float)
    n_ports = int(round(np.sqrt((arr.shape[1] - 1) / 2)))
    if 1 + 2 * n_ports * n_ports != arr.shape[1]:
        raise ValueError(
            f"列数 {arr.shape[1]} 对不上端口数：推出来是 {n_ports} 端口"
        )
    freq_hz = arr[:, 0] * freq_scale
    s: dict[tuple[int, int], np.ndarray] = {}
    def to_complex(re: np.ndarray, im: np.ndarray) -> np.ndarray:
        if fmt == "RI":
            return re + 1j * im
        if fmt == "MA":
            return im * np.exp(1j * np.deg2rad(re))
        if fmt == "DB":
            return (10 ** (re / 20.0)) * np.exp(1j * np.deg2rad(im))
        raise ValueError(f"不认识的格式：{fmt}")
    col = 1
    for i in range(1, n_ports + 1):
        for j in range(1, n_ports + 1):
            s[(i, j)] = to_complex(arr[:, col], arr[:, col + 1])
            col += 2
    return Touchstone(
        path=path, freq_hz=freq_hz, s=s, z0=z0, fmt=fmt, n_ports=n_ports
    )
# ---------------------------------------------------------------- 检查
def check(ts: Touchstone, expected_points: int | None = None) -> CheckResult:
    res = CheckResult(ok=True)
    f = ts.freq_hz
    if len(f) < 2:
        res.ok = False
        res.issues.append("频点少于 2 个")
        return res
    if not np.all(np.diff(f) > 0):
        res.ok = False
        bad = int(np.argmax(np.diff(f) <= 0)) + 1
        res.issues.append(f"频率不是严格递增，第 {bad} 个点开始出问题")
    else:
        res.notes.append(f"频率严格递增，{f[0]/1e9:g} → {f[-1]/1e9:g} GHz")
    step = float(np.mean(np.diff(f)))
    res.notes.append(f"平均步长 {step/1e6:.3g} MHz，共 {len(f)} 点")
    for name, data in ts.s.items():
        if np.isnan(data).any():
            res.ok = False
            res.issues.append(f"S{name[0]}{name[1]} 里有 NaN")
        if np.isinf(data).any():
            res.ok = False
            res.issues.append(f"S{name[0]}{name[1]} 里有无穷大")
    if expected_points is not None and len(f) != expected_points:
        res.ok = False
        res.issues.append(f"频点数是 {len(f)}，和约定的 {expected_points} 不一致")
    # 和 config 的网格对比（同一批数据应该用同一个扫频）
    grid = config.FREQ_GRID
    if len(grid) == len(f) and np.allclose(grid, f, rtol=1e-9, atol=1.0):
        res.notes.append("扫频和 config.FREQ_GRID 完全一致")
    else:
        res.notes.append(
            f"扫频和 config.FREQ_GRID 不一致（文件 {len(f)} 点，config {len(grid)} 点），"
            f"后面比较前要先对齐"
        )
    # 互易性：无源网络一般 S12 = S21
    if (1, 2) in ts.s and (2, 1) in ts.s:
        if np.allclose(ts.s[(1, 2)], ts.s[(2, 1)], atol=1e-9):
            res.notes.append("S12 == S21（互易，符合无源网络）")
        else:
            res.notes.append("S12 != S21（结构非互易，或数据有问题）")
    return res
# ---------------------------------------------------------------- 转表
def to_long_table(ts: Touchstone,
                  candidate_id: str,
                  params: dict[str, float] | None = None) -> pd.DataFrame:
    """长表：每一行是"某组参数、某个频率、一串 S 参数"。"""
    cols: dict[str, np.ndarray] = {
        "candidate_id": np.full(len(ts.freq_hz), str(candidate_id), dtype=object),
        "freq_hz": ts.freq_hz,
    }
    for key, val in (params or {}).items():
        cols[key] = np.full(len(ts.freq_hz), float(val))
    for (i, j), data in sorted(ts.s.items()):
        cols[f"s{i}{j}_re"] = data.real
        cols[f"s{i}{j}_im"] = data.imag
    return pd.DataFrame(cols)
# ---------------------------------------------------------------- 命令行
def _find_default_file() -> Path:
    candidates = sorted(config.DATA_RAW.glob("*.s*p"))
    if not candidates:
        raise FileNotFoundError(
            f"{config.DATA_RAW} 里没有 .s2p 文件。用 --file 指定路径。"
        )
    return candidates[0]
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读 Touchstone 文件并检查")
    parser.add_argument("--file", type=str, default=None, help="文件路径")
    parser.add_argument("--candidate-id", type=str, default="001", help="这组参数的编号")
    parser.add_argument("--save", action="store_true", help="转成长表存到 processed 目录")
    args = parser.parse_args(argv)
    try:
        path = Path(args.file) if args.file else _find_default_file()
        ts = read_touchstone(path)
        res = check(ts)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    print(f"文件：{ts.path.name}")
    print(f"端口：{ts.n_ports}    格式：{ts.fmt}    参考阻抗：{ts.z0:g} Ω")
    print(f"包含：{', '.join(ts.matrix_names()).upper()}")
    print()
    print("检查：")
    print(res.format())
    if args.save:
        df = to_long_table(ts, candidate_id=args.candidate_id)
        out = config.DATA_PROCESSED / f"{ts.path.stem}_long.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print(f"长表已保存：{out}")
        print(f"  行数 {len(df)}，列：{', '.join(df.columns[:9])}...")
    return 0 if res.ok else 2
if __name__ == "__main__":
    raise SystemExit(main())