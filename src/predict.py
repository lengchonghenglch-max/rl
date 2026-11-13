r"""
预测：把参数组合喂给 GP，拿到四项指标和不确定度
================================================
gp_model 负责训练，predict 负责"拿来就用"：
    - 预测单个组合
    - 预测一张候选表（acquisition 生成的）
    - 输出成 csv，给下一步选点用
放哪里：B_Group/src/predict.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
预测一组：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.predict --c1 0.8 --c2 1.2
预测整张候选表：
    python -m src.predict --input outputs\candidates候选参数\initial_100.csv
常见报错：
    [失败] 模型文件不存在
        → 先跑 python -m src.gp_model --save
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
try:
    from . import config
    from .gp_model import TARGETS, GPModel
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from gp_model import TARGETS, GPModel  # type: ignore
MODEL_PATH = config.MODEL_DIR / "gp_model.joblib"
# 指标方向：这四项都是"越小越好"（插损小、回波负得多、阻带衰减负得多）
LOWER_IS_BETTER = {"il_db": True, "s11_db": False, "s22_db": False, "stop_db": False}
def margin(values: dict[str, float]) -> float:
    """所有指标里最紧的那一条还剩多少余量（负 = 不合格）。
    归一化到 0 附近，方便不同量纲的指标放在一起比。
    """
    limits = {m.key: m.limit_db for m in config.METRICS}
    # 四项都是"不超过阈值"：余量 = 阈值 - 实际值，再按阈值归一化
    margins = [(limits[k] - v) / abs(limits[k]) for k, v in values.items()]
    return float(min(margins))
def predict_frame(model: GPModel, df: pd.DataFrame) -> pd.DataFrame:
    """对一张含 c1_pf / c2_pf 的表做批量预测。"""
    rows = []
    for _, r in df.iterrows():
        out = model.predict(float(r["c1_pf"]), float(r["c2_pf"])).iloc[0].to_dict()
        out["margin"] = margin({k: out[k] for k in TARGETS})
        rows.append(out)
    return pd.DataFrame(rows)
def load_model(path: Path = MODEL_PATH) -> GPModel:
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}，先跑 python -m src.gp_model --save")
    return GPModel.load(path)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用已训练的 GP 做预测")
    parser.add_argument("--c1", type=float, help="C1（pF）")
    parser.add_argument("--c2", type=float, help="C2（pF）")
    parser.add_argument("--input", type=str, help="候选表 csv")
    parser.add_argument("--out", type=str, default=None, help="输出 csv")
    args = parser.parse_args(argv)
    try:
        model = load_model()
    except FileNotFoundError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    if args.c1 is not None and args.c2 is not None:
        out = model.predict(args.c1, args.c2)
        out["margin"] = margin({k: out.iloc[0][k] for k in TARGETS})
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return 0
    if args.input:
        df = pd.read_csv(args.input)
        missing = {"c1_pf", "c2_pf"} - set(df.columns)
        if missing:
            print(f"[失败] 候选表缺列：{missing}", file=sys.stderr)
            return 1
        pred = predict_frame(model, df)
        out_path = Path(args.out) if args.out else (
            config.OUT_PREDICTIONS / f"{Path(args.input).stem}_pred.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"预测 {len(pred)} 组，已保存：{out_path}")
        print(pred.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return 0
    parser.print_help()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

