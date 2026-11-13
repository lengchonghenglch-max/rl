r"""
GP 代理模型：训练、预测、评估、保存
====================================
输入：一组参数（C1、C2）→ 输出：四项指标（插损 / S11 / S22 / 阻带）
用 scikit-learn 的 GaussianProcessRegressor，每个指标一个模型，同时给出预测值和不确定度。
放哪里：B_Group/src/gp_model.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
用 data\raw原数据 下的文件夹训练（每个文件夹一组参数）：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.gp_model
指定数据目录：
    python -m src.gp_model --data "D:\某个\数据文件夹"
交叉验证 + 保存模型：
    python -m src.gp_model --cv --save
预测单个组合：
    python -m src.gp_model --predict 0.8 1.2
常见报错：
    ModuleNotFoundError: No module named 'sklearn'
        → pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
    ValueError: 没有找到任何仿真数据
        → --data 指错了，或者文件夹里没有 .s2p
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
try:
    from . import config
    from .data_process import read_touchstone
    from .metrics import compute
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from data_process import read_touchstone  # type: ignore
    from metrics import compute  # type: ignore
TARGETS = ["il_db", "s11_db", "s22_db", "stop_db"]
# ---------------------------------------------------------------- 建训练集
def _parse_params(name: str) -> tuple[float, float] | None:
    """从文件夹名解析 C1、C2。支持 'c1 0.3 c2 0.9' 和 'c1 c2 0.5'。"""
    m = re.search(r"c1\s+([\d.]+)\s+c2\s+([\d.]+)", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"c1\s+c2\s+([\d.]+)", name)
    if m:
        v = float(m.group(1))
        return v, v
    return None
def scan_dataset(root: str | Path) -> pd.DataFrame:
    """扫一个目录，每个子文件夹算一组数据（参数 + 四项指标）。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        params = _parse_params(folder.name)
        if params is None:
            continue
        files = sorted(folder.glob("*.s2p"))
        if not files:
            continue
        ts = read_touchstone(files[0])
        s = ts.s
        report = compute(ts.freq_hz, s[(1, 1)], s[(2, 1)], s[(2, 2)])
        values = report.to_dict()
        rows.append({
            "c1_pf": params[0],
            "c2_pf": params[1],
            "source": str(files[0]),
            **{k: values[k] for k in TARGETS},
            "passed": report.passed,
        })
    if not rows:
        raise ValueError(
            f"在 {root} 下没有找到任何仿真数据。"
            f"文件夹名要能解析出 c1/c2，例如 'c1 0.3 c2 0.9'。"
        )
    return pd.DataFrame(rows)
# ---------------------------------------------------------------- 模型
class GPModel:
    """四个指标各一个 GP；输入标准化，输出由 GP 内部标准化。"""
    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.models: dict[str, GaussianProcessRegressor] = {}
        self.feature_names = ["c1_pf", "c2_pf"]
    @staticmethod
    def _kernel() -> object:
        # 各向同性核：C1 与 C2 对指标的作用近似对称，共用一个长度尺度更稳
        return (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(5e-2, 5e1))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        )
    def fit(self, df: pd.DataFrame) -> "GPModel":
        X = self.x_scaler.fit_transform(df[self.feature_names].to_numpy(float))
        for target in TARGETS:
            y = df[target].to_numpy(float)
            gp = GaussianProcessRegressor(
                kernel=self._kernel(),
                normalize_y=True,
                n_restarts_optimizer=3,
                random_state=self.random_state,
            )
            gp.fit(X, y)
            self.models[target] = gp
        return self
    def predict(self, c1: float, c2: float) -> pd.DataFrame:
        X = self.x_scaler.transform(np.array([[c1, c2]], float))
        out = {"c1_pf": c1, "c2_pf": c2}
        for target, gp in self.models.items():
            mu, sd = gp.predict(X, return_std=True)
            out[target] = float(mu[0])
            out[f"{target}_sd"] = float(sd[0])
        return pd.DataFrame([out])
    def predict_grid(self, n: int = 15) -> pd.DataFrame:
        vals = np.linspace(0.1, 1.5, n)
        pts = np.array([[a, b] for a in vals for b in vals])
        X = self.x_scaler.transform(pts)
        out = pd.DataFrame(pts, columns=["c1_pf", "c2_pf"])
        for target, gp in self.models.items():
            mu, sd = gp.predict(X, return_std=True)
            out[target] = mu
            out[f"{target}_sd"] = sd
        return out
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path
    @staticmethod
    def load(path: str | Path) -> "GPModel":
        import joblib
        return joblib.load(path)
# ---------------------------------------------------------------- 评估
def cross_validate(df: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """按设计点划分（每个点整条留出），报告每个指标的 RMSE / MAE。"""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    pred = np.full((len(df), len(TARGETS)), np.nan)
    sd = np.full((len(df), len(TARGETS)), np.nan)
    for tr, te in kf.split(df):
        m = GPModel().fit(df.iloc[tr])
        for j, t in enumerate(TARGETS):
            X = m.x_scaler.transform(df.iloc[te][m.feature_names].to_numpy(float))
            mu, s = m.models[t].predict(X, return_std=True)
            pred[te, j] = mu
            sd[te, j] = s
    rows = []
    for j, t in enumerate(TARGETS):
        y = df[t].to_numpy(float)
        e = pred[:, j] - y
        rows.append({
            "指标": t,
            "RMSE": float(np.sqrt(np.nanmean(e ** 2))),
            "MAE": float(np.nanmean(np.abs(e))),
            "平均不确定度": float(np.nanmean(sd[:, j])),
        })
    return pd.DataFrame(rows)
# ---------------------------------------------------------------- 命令行
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GP 代理模型")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--cv", action="store_true", help="做交叉验证")
    parser.add_argument("--save", action="store_true", help="保存模型与预测网格")
    parser.add_argument("--predict", nargs=2, type=float, metavar=("C1", "C2"),
                        help="预测单个组合")
    args = parser.parse_args(argv)
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    try:
        df = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    print(f"数据：{len(df)} 组，来自 {data_dir}")
    print(f"  C1 {df.c1_pf.min():g} – {df.c1_pf.max():g} pF，"
          f"C2 {df.c2_pf.min():g} – {df.c2_pf.max():g} pF")
    print(f"  合格 {int(df.passed.sum())} 组 / 不合格 {int((~df.passed).sum())} 组")
    if args.predict:
        model = GPModel().fit(df)
        print()
        print(model.predict(*args.predict).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return 0
    if args.cv:
        print()
        print("五折交叉验证（按设计点划分）")
        print(cross_validate(df).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    model = GPModel().fit(df)
    for t, gp in model.models.items():
        print(f"  {t:<8} 核：{gp.kernel_}")
    if args.save:
        p = model.save(config.MODEL_DIR / "gp_model.joblib")
        grid = model.predict_grid()
        gpath = config.OUT_PREDICTIONS / "prediction_grid.csv"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(gpath, index=False, encoding="utf-8-sig")
        print(f"\n模型已保存：{p}")
        print(f"预测网格已保存：{gpath}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

