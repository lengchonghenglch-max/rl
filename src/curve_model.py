r"""
曲线模型：预测整条 S 参数曲线，而不只是四个数
==============================================

做法（降维 + GP）：

    每条曲线 159 个点 → 看成一根向量
    29 条曲线做 PCA，保留前 k 个主成分（通常能解释 99% 以上变化）
    对每个主成分的系数，各训一个 GP：C1, C2 → 系数
    预测时：GP 给系数 → 乘回主成分 → 还原整条曲线

不确定度：每个系数的 GP 标准差按载荷平方和传播到曲线上，得到 ±1σ 的带。

S11 / S21 / S22 各建一套。

放哪里：B_Group/src/curve_model.py

------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
看某组参数的预测曲线：

    cd /d D:\rf\挑战杯\B_Group
    python -m src.curve_model --c1 0.8 --c2 1.2

做留一法检验（每次留一条曲线，用其余 28 条预测它）：

    python -m src.curve_model --loo

常见报错：

    ValueError: 样本太少
        → 至少要 3 组数据才能做 PCA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

try:
    from . import config
    from .data_process import read_touchstone
    from .gp_model import scan_dataset
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from data_process import read_touchstone  # type: ignore
    from gp_model import scan_dataset  # type: ignore

PARAMS = ("s11", "s21", "s22")


def _db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def load_curves(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """读每条曲线，返回公共频率轴和三个参数的 dB 矩阵（样本 × 频点）。"""
    freqs = None
    data: dict[str, list[np.ndarray]] = {p: [] for p in PARAMS}
    for _, row in df.iterrows():
        ts = read_touchstone(row["source"])
        if freqs is None:
            freqs = ts.freq_hz
        elif len(ts.freq_hz) != len(freqs) or not np.allclose(ts.freq_hz, freqs, atol=1.0):
            raise ValueError("各文件的频率轴不一致，先统一到同一套扫频再建模")
        data["s11"].append(_db(ts.s[(1, 1)]))
        data["s21"].append(_db(ts.s[(2, 1)]))
        data["s22"].append(_db(ts.s[(2, 2)]))
    return freqs, {k: np.vstack(v) for k, v in data.items()}


class CurveModel:
    """每个 S 参数一套 PCA + GP。"""

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.parts: dict[str, dict] = {}

    @staticmethod
    def _kernel():
        return (ConstantKernel(1.0, (1e-2, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(5e-2, 5e1))
                + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1)))

    def fit(self, df: pd.DataFrame) -> "CurveModel":
        freqs, curves = load_curves(df)
        X = self.scaler.fit_transform(df[["c1_pf", "c2_pf"]].to_numpy(float))
        self.freq_hz = freqs

        n = len(df)
        k = min(self.n_components, n - 1)
        if k < 2:
            raise ValueError(f"样本太少（{n} 组），至少要 3 组才能做 PCA")

        for p in PARAMS:
            Y = curves[p]
            pca = PCA(n_components=k, random_state=0)
            scores = pca.fit_transform(Y)          # 样本 × k
            gps = []
            for j in range(k):
                gp = GaussianProcessRegressor(
                    kernel=self._kernel(), normalize_y=True,
                    n_restarts_optimizer=3, random_state=0)
                gp.fit(X, scores[:, j])
                gps.append(gp)
            self.parts[p] = {
                "pca": pca, "gps": gps, "k": k,
                "evr": float(pca.explained_variance_ratio_.sum()),
            }
        return self

    def predict(self, c1: float, c2: float) -> dict[str, dict[str, np.ndarray]]:
        X = self.scaler.transform(np.array([[c1, c2]], float))
        out = {}
        for p, part in self.parts.items():
            pca, gps = part["pca"], part["gps"]
            scores, sds = [], []
            for gp in gps:
                mu, sd = gp.predict(X, return_std=True)
                scores.append(mu[0])
                sds.append(sd[0])
            scores = np.array(scores)
            sds = np.array(sds)
            mean = pca.mean_ + scores @ pca.components_
            var = ((sds[:, None] * pca.components_) ** 2).sum(0)
            out[p] = {"mean_db": mean, "sd_db": np.sqrt(var), "freq_hz": self.freq_hz}
        return out


def leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """每次留一组，用其余数据预测它的曲线，统计误差。"""
    rows = []
    for i in range(len(df)):
        train = df.drop(index=df.index[i])
        test = df.iloc[i]
        m = CurveModel().fit(train)
        pred = m.predict(test.c1_pf, test.c2_pf)
        _, curves = load_curves(df.iloc[[i]])
        for p in PARAMS:
            err = pred[p]["mean_db"] - curves[p][0]
            rows.append({
                "c1_pf": test.c1_pf, "c2_pf": test.c2_pf, "参数": p,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "最大误差": float(np.max(np.abs(err))),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S 参数曲线模型")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--c1", type=float, help="预测这组（C1）")
    parser.add_argument("--c2", type=float, help="预测这组（C2）")
    parser.add_argument("--loo", action="store_true", help="做留一法检验")
    args = parser.parse_args(argv)

    data_dir = Path(args.data) if args.data else config.DATA_RAW
    try:
        df = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    print(f"数据 {len(df)} 组")

    if args.loo:
        print("留一法检验中（每次重建一次模型，稍慢）…")
        res = leave_one_out(df)
        print(res.groupby("参数")[["MAE", "RMSE", "最大误差"]].mean().round(3).to_string())
        worst = res.sort_values("MAE", ascending=False).head(5)
        print("\n误差最大的 5 组：")
        print(worst.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        return 0

    if args.c1 is None or args.c2 is None:
        parser.print_help()
        return 0

    model = CurveModel().fit(df)
    for p, part in model.parts.items():
        print(f"  {p}: 主成分 {part['k']} 个，累计解释 {part['evr']:.4%}")
    out = model.predict(args.c1, args.c2)
    for p in PARAMS:
        mean = out[p]["mean_db"]
        print(f"  {p}: 全频段 {mean.min():.1f} ~ {mean.max():.1f} dB，"
              f"平均不确定度 {out[p]['sd_db'].mean():.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
