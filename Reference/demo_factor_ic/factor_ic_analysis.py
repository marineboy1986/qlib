"""
==============================================================================
因子 IC / ICIR 分析 Demo — 基于 Qlib 数据 API
==============================================================================
演示如何用 Qlib 数据 + Python 计算因子的 IC (Information Coefficient)
和 ICIR (IC Information Ratio)，评估因子的预测能力。

核心指标说明:
  - IC (Information Coefficient): 因子值与下期收益的 Spearman 秩相关系数
    → 衡量因子是否具备选股能力
  - ICIR (IC Information Ratio): IC 均值 / IC 标准差
    → 衡量因子预测能力的稳定性 (ICIR > 0.5 为优秀)
  - IC 胜率: IC > 0 的期数占比

运行环境: pip install pyqlib pandas numpy matplotlib
==============================================================================
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ───────────────────────────── 1. Qlib 初始化 ─────────────────────────────
sys.path = [p for p in sys.path if "/csai-workspace/qlib" not in p]
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH

DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
qlib.init(provider_uri=DATA_DIR, region=REG_CN)

print("=" * 60)
print("  因子 IC / ICIR 分析 Demo")
print(f"  Qlib {qlib.__version__} | 数据: {DATA_DIR}")
print("=" * 60)

# ═══════════════════════════ 2. 加载数据 ═══════════════════════════
# 股票池: CSI300
# 时间: 2018-01-01 ~ 2020-08-01
MARKET = "csi300"
START  = "2018-01-01"
END    = "2020-08-01"
FREQ   = "day"

print(f"\n[1/4] 加载 CSI300 成分股 ({START} ~ {END})...")
inst_file = os.path.expanduser(f"~/.qlib/qlib_data/cn_data/instruments/{MARKET}.txt")
stock_list = sorted(set(
    line.strip().split("\t")[0] for line in open(inst_file) if line.strip()
))
print(f"      成分股: {len(stock_list)} 只")

# ═══════════════════════════ 3. 定义待测试因子 ═══════════════════════════
# 使用 Qlib 内置表达式定义几个常见因子
FACTOR_DEFS = {
    "ROC20_LAG": "$close / Ref($close, 20) - 1",             # 过去20日涨跌幅 (反转因子)
    "ROC5_LAG":  "$close / Ref($close, 5) - 1",              # 过去5日涨跌幅 (反转因子)
    "VOLATILITY20": "Std($close, 20)",                       # 20日波动率
    "VOLUME_RATIO": "$volume / Mean($volume, 20)",           # 20日量比
    "HIGH_LOW":     "($high - $low) / $close",               # 日内振幅
    "AMOUNT_MA5":   "$amount / Mean($amount, 5)",            # 5日金额比
}

print(f"\n[2/4] 定义 {len(FACTOR_DEFS)} 个待测试因子:")
for name, expr in FACTOR_DEFS.items():
    print(f"      {name:>15s} = {expr}")

# 分批量加载因子数据 + 基础行情 (用于算收益)
factor_names = list(FACTOR_DEFS.keys())
factor_exprs = list(FACTOR_DEFS.values())

# 还需要收盘价来计算未来收益
all_fields = ["$close"] + factor_exprs

print(f"\n      分批加载因子数据...")
raw_parts = []
for i in range(0, len(stock_list), 100):
    batch = stock_list[i:i+100]
    df = D.features(batch, all_fields, START, END, freq=FREQ)
    raw_parts.append(df)
raw = pd.concat(raw_parts)
print(f"      完成 | 样本: {len(raw)} 条")

# 处理列索引
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(0)

# 重命名
col_map = {"$close": "Close"}
for expr, name in zip(factor_exprs, factor_names):
    col_map[expr] = name
raw = raw.rename(columns=col_map)
print(f"      列: {list(raw.columns)}")

# ═══════════════════════════ 4. 计算 IC / ICIR ═══════════════════════════
print(f"\n[3/4] 计算 Rank IC & ICIR...")

def calc_rank_ic(factor_series, forward_ret_series):
    """计算单日横截面 Rank IC"""
    df = pd.DataFrame({"factor": factor_series, "ret": forward_ret_series}).dropna()
    if len(df) < 10:
        return np.nan
    return spearmanr(df["factor"], df["ret"])[0]

def calc_icir(ic_series):
    """计算 ICIR = IC均值 / IC标准差"""
    ic_s = ic_series.dropna()
    if len(ic_s) < 5:
        return np.nan
    return ic_s.mean() / ic_s.std()

# 计算每只股票的未来 5 日收益 (T+5 相对于 T 的收益)
raw["RET5"] = raw.groupby("instrument")["Close"].transform(
    lambda x: x.shift(-5) / x - 1
)

# 对每个日期、每个因子计算 Rank IC
results = []
dates = sorted(raw.index.get_level_values("datetime").unique())

for d in dates:
    day_data = raw[raw.index.get_level_values("datetime") == d]
    for fname in factor_names:
        ic = calc_rank_ic(day_data[fname], day_data["RET5"])
        results.append({"date": d, "factor": fname, "IC": ic})

ic_df = pd.DataFrame(results).dropna(subset=["IC"])

# 汇总统计
summary = []
for fname in factor_names:
    f_ic = ic_df[ic_df["factor"] == fname]["IC"]
    if len(f_ic) < 5:
        continue
    ic_mean = f_ic.mean()
    ic_std  = f_ic.std()
    icir    = ic_mean / ic_std if ic_std != 0 else 0
    win_rate = (f_ic > 0).mean()

    # 状态判定
    if abs(ic_mean) > 0.03 and icir > 0.5:
        status = "🟢 强势"
    elif abs(ic_mean) > 0.02 and icir > 0.3:
        status = "🟡 观察"
    elif abs(ic_mean) < 0.01 or icir < 0.2:
        status = "🔴 失效"
    else:
        status = "🟡 观察"

    summary.append({
        "因子": fname,
        "IC均值": ic_mean,
        "IC标准差": ic_std,
        "ICIR": icir,
        "IC胜率": win_rate,
        "状态": status,
    })

summary_df = pd.DataFrame(summary)
summary_df["|ICIR|"] = summary_df["ICIR"].abs()
summary_df = summary_df.sort_values("|ICIR|", ascending=False).drop(columns="|ICIR|")

print(f"\n  ★ 因子 IC / ICIR 评估结果")
print(f"  {'─'*70}")
print(f"  {'因子名':>15s}  {'IC均值':>8s}  {'ICIR':>7s}  {'IC胜率':>8s}  {'状态':>10s}")
print(f"  {'─'*70}")
for _, row in summary_df.iterrows():
    print(f"  {row['因子']:>15s}  {row['IC均值']:>+8.4f}  {row['ICIR']:>+7.2f}  {row['IC胜率']:>7.1%}  {row['状态']}")

# ═══════════════════════════ 5. IC 时序分析 ═══════════════════════════
print(f"\n[4/4] IC 时序分解...")

for fname in factor_names:
    f_ic = ic_df[ic_df["factor"] == fname].copy()
    if len(f_ic) < 20:
        continue

    # 滚动 ICIR (60 日窗口)
    f_ic = f_ic.sort_values("date")
    f_ic["ICIR_60D"] = f_ic["IC"].rolling(60).apply(
        lambda x: x.mean() / x.std() if x.std() != 0 else 0
    )

    # 最近 N 日 IC 均值
    for window in [20, 60, 120]:
        col = f"IC_{window}D_MA"
        f_ic[col] = f_ic["IC"].rolling(window).mean()

    print(f"\n  ── {fname} ──")
    print(f"     整体  ICIR = {f_ic['IC'].mean() / f_ic['IC'].std():+.2f}")
    for w in [20, 60, 120]:
        col = f"IC_{w}D_MA"
        latest = f_ic[col].dropna().iloc[-1] if len(f_ic) >= w else np.nan
        print(f"     近{w:>3d}日 IC 均值: {latest:+.4f}" if not np.isnan(latest) else f"     近{w:>3d}日 IC 均值: N/A")

# ═══════════════════════════ 6. 分组收益分析 (仅对最佳因子) ═══════════════════════════
best_factor = summary_df.iloc[0]["因子"]
print(f"\n\n  ★ 分组收益分析 — 最佳因子: {best_factor}")
print(f"  {'─'*50}")

def calc_group_returns(factor_name, n_groups=5):
    """按因子值分为 n 组, 计算每组下期平均收益"""
    grp_list = []
    for d in dates:
        day_data = raw[raw.index.get_level_values("datetime") == d][[factor_name, "RET5"]].dropna()
        if len(day_data) < n_groups * 5:
            continue
        day_data = day_data.copy()
        day_data["group"] = pd.qcut(day_data[factor_name].rank(method="first"),
                                      q=n_groups, labels=list(range(n_groups, 0, -1)))
        # group 5 = 因子值最高, group 1 = 因子值最低
        if factor_name in ["ROC20", "ROC5"]:
            # 反转因子: 因子值低(跌得多) → 预期收益高, 所以 group 1 应该是多
            day_data["group"] = pd.qcut(day_data[factor_name].rank(method="first"),
                                          q=n_groups, labels=list(range(1, n_groups+1)))
        grp_mean = day_data.groupby("group")["RET5"].mean()
        grp_list.append(grp_mean)

    result = pd.DataFrame(grp_list)
    return result

grp_returns = calc_group_returns(best_factor, n_groups=5)
print(f"\n  {'组别':>8s}  {'平均收益':>10s}  {'年化收益':>10s}")
print(f"  {'─'*32}")
for g in sorted(grp_returns.columns):
    avg = grp_returns[g].mean()
    ann = (1 + avg) ** 252 - 1
    print(f"  {'第'+str(g)+'组':>8s}  {avg:>+9.4f}  {ann:>+9.2%}")

# 多空收益
ls_return = grp_returns[max(grp_returns.columns)] - grp_returns[min(grp_returns.columns)]
ls_mean = ls_return.mean()
ls_ann = (1 + ls_mean) ** 252 - 1
ls_ir = ls_return.mean() / ls_return.std() * np.sqrt(252)
print(f"  {'─'*32}")
print(f"  {'多空收益':>8s}  {ls_mean:>+9.4f}  {ls_ann:>+9.2%}")
print(f"  {'多空IR':>8s}  {ls_ir:>+9.2f}")

# ═══════════════════════════ 7. 汇总报告 ═══════════════════════════
print(f"\n{'='*60}")
print(f"  📊 因子 IC/ICIR 分析报告")
print(f"{'='*60}")
print(f"""
  测试区间: {START} ~ {END}
  股票池:   CSI300 ({len(stock_list)} 只)
  收益频率: 5 日 (T+5)
  总样本数: {len(raw):,} 条

  ★ 因子总评:
""")
for _, row in summary_df.iterrows():
    print(f"    {row['因子']:>15s}  IC={row['IC均值']:+.4f}  ICIR={row['ICIR']:+.2f}  "
          f"胜率={row['IC胜率']:.0%}  {row['状态']}")

print(f"""
  ★ 最佳因子: {best_factor}
    → 多空年化收益: {(1+ls_mean)**252-1:+.2%}
    → 多空 IR: {ls_ir:.2f}

  ★ 解读:
    - ICIR > 0.5: 因子预测能力强且稳定 (🟢)
    - ICIR 0.3~0.5: 有预测能力但波动较大 (🟡)
    - ICIR < 0.2 或 IC 均值接近 0: 因子失效 (🔴)
    - 多空收益显著 + 分组单调 = 因子可用
""")
print(f"  ✅ 分析完成!")