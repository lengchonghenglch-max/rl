r"""
B 组主程序：一条命令跑完整条线
================================
    [1] 读配置
    [2] 检查缺项
    [3] 扫描仿真数据（每个文件夹一组参数）
    [4] 指标统计（合格几组、卡在哪条）
    [5] 训练 GP + 交叉验证
    [6] 选下一批候选（UCB）
    [7] 安全检查
    [8] 保存结果
参数域没填、数据没到时，每一步会自己说明为什么跳过，不会中断。
放哪里：B_Group/main.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
    cd /d D:\rf\挑战杯\B_Group
    python main.py
指定数据目录、想要几个候选：
    python main.py --data "D:\某目录" --n 5
只跑自检（不读数据）：
    python main.py --demo
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src import config
from src.acquisition import pick, rank
from src.gp_model import GPModel, cross_validate, scan_dataset
from src.safety_check import OK, check_frame
from src.search_space import IncompleteRangeError, check  # noqa: F401
STEP = 0
def step(title: str) -> None:
    global STEP
    STEP += 1
    print(f"\n[{STEP}] {title}")
    print("-" * 62)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B 组主程序")
    parser.add_argument("--data", type=str, default=None, help="仿真数据目录")
    parser.add_argument("--n", type=int, default=5, help="推荐几组候选")
    parser.add_argument("--demo", action="store_true", help="只做自检，不读数据")
    parser.add_argument("--no-fig", action="store_true", help="不生成图表")
    parser.add_argument("--no-exp", action="store_true", help="不跑对比实验")
    args = parser.parse_args(argv)
    step("读配置")
    print(config.summary(show_missing=False))
    step("检查缺项")
    missing = config.missing_items()
    if missing:
        for m in missing:
            print(f"  - {m}")
    else:
        print("  配置齐全")
    data_dir = Path(args.data) if args.data else config.DATA_RAW
    step(f"扫描仿真数据：{data_dir}")
    try:
        df = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [跳过] {exc}")
        print("  把仿真结果按 'c1 0.3 c2 0.9' 的文件夹名放进去即可。")
        return 1
    print(f"  {len(df)} 组设计，C1 {df.c1_pf.min():g}–{df.c1_pf.max():g} pF，"
          f"C2 {df.c2_pf.min():g}–{df.c2_pf.max():g} pF")
    diag = int((np.isclose(df.c1_pf, df.c2_pf)).sum())
    print(f"  其中 C1=C2 的 {diag} 组，C1≠C2 的 {len(df) - diag} 组")
    step("指标统计")
    ok = int(df.passed.sum())
    print(f"  合格 {ok} 组，不合格 {len(df) - ok} 组")
    for m in config.METRICS:
        bad = int(((df[m.key] > m.limit_db) if m.direction == "max"
                   else (df[m.key] < m.limit_db)).sum())
        print(f"    {m.name}：不满足 {bad} 组")
    step("训练 GP + 交叉验证")
    model = GPModel().fit(df)
    print(cross_validate(df).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    model.save(config.MODEL_DIR / "gp_model.joblib")
    print(f"  模型已保存：{config.MODEL_DIR / 'gp_model.joblib'}")
    step(f"选下一批候选（{args.n} 组）")
    grid = [[a, b] for a in np.linspace(df.c1_pf.min(), df.c1_pf.max(), 40)
            for b in np.linspace(df.c2_pf.min(), df.c2_pf.max(), 40)]
    cand = pd.DataFrame(grid, columns=["c1_pf", "c2_pf"])
    picked = pick(rank(model, cand, beta=0.5), args.n, 0.05)
    print(picked[["c1_pf", "c2_pf", "margin", "ucb"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    step("安全检查")
    safety = check_frame(picked, df)
    print(safety.to_string(index=False))
    for verdict in (OK, "必须仿真", "禁止"):
        print(f"  {verdict}：{int((safety['结论'] == verdict).sum())} 组")
    step("保存")
    out_rec = config.OUT_RECOMMENDATIONS / "next_batch.csv"
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    merged = picked.merge(safety[["c1_pf", "c2_pf", "结论", "理由"]], on=["c1_pf", "c2_pf"])
    merged.to_csv(out_rec, index=False, encoding="utf-8-sig")
    print(f"  推荐表：{out_rec}")
    grid_out = config.OUT_PREDICTIONS / "prediction_grid.csv"
    grid_out.parent.mkdir(parents=True, exist_ok=True)
    model.predict_grid().to_csv(grid_out, index=False, encoding="utf-8-sig")
    print(f"  预测网格：{grid_out}")
    if not args.no_fig:
        step("生成图表")
        try:
            from src.visualize import fig_s_full, fig_s_passband, fig_s_stopband
            fig_dir = config.ROOT / "outputs" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for fn in (fig_s_full, fig_s_passband, fig_s_stopband):
                print(f"  {fn(df, fig_dir)}")
        except Exception as exc:  # 画图失败不影响主流程
            print(f"  [跳过] 出图失败：{type(exc).__name__}: {exc}")
            print("  多半是 matplotlib 没装：pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
    if not args.no_exp:
        step("对比实验（随机 / 网格 / 贝叶斯）")
        try:
            from src.experiment import main as run_experiment
            run_experiment(["--data", str(data_dir),
                            "--out", str(config.ROOT / "outputs" / "figures")])
        except Exception as exc:  # 实验失败不影响主流程
            print(f"  [跳过] 实验失败：{type(exc).__name__}: {exc}")
    print("\n下一步：把推荐表交给仿真组；结果回来后放进数据目录，重跑本程序。")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
