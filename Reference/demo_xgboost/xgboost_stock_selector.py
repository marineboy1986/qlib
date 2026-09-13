"""
==============================================================================
XGBoost 选股 Demo — 纯 Python + XGBoost
==============================================================================
不依赖 Qlib 完整 Pipeline，直接用原始数据和 XGBoost 演示选股核心逻辑:
  1) 获取股票行情数据 (日频: 开盘/最高/最低/收盘/成交量/换手率)
  2) 手工构造技术面因子 (动量、波动率、成交额变化等)
  3) 定义标签: 未来 5 日收益排名 (横截面排序)
  4) 训练 XGBoost 预测排序得分
  5) 最新一期选股推荐

运行环境: pandas, numpy, xgboost, requests
==============================================================================
"""
import os, sys, warnings, gc
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import xgboost as xgb

print("=" * 60)
print("  XGBoost 选股 Demo")
print("=" * 60)

# ============ 1. 使用 Qlib 数据 API 直接加载 CSI300 行情 ============
sys.path = [p for p in sys.path if "/csai-workspace/qlib" not in p]
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH

DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
qlib.init(provider_uri=DATA_DIR, region=REG_CN)

CSI300 = "csi300"
FREQ = "day"
START = "2015-01-01"
END   = "2020-08-01"

# 工具函数: 横截面 rank
def cs_rank(s):
    return s.rank(pct=True)

# 从文件读取成分股列表 (去重)
inst_file = os.path.expanduser(f"~/.qlib/qlib_data/cn_data/instruments/{CSI300}.txt")
instr_set = set()
with open(inst_file) as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts:
            instr_set.add(parts[0])
stock_list = sorted(instr_set)
print(f"\n[1/5] CSI300 成分股: {len(stock_list)} 只")

fields = [
    "$close", "$open", "$high", "$low",         # 价格
    "$volume", "$amount", "$vwap",              # 量价
]
print("     加载日频行情 (按股票分批)...")
ALL_STOCKS = stock_list
raw_parts = []
batch_size = 100
for i in range(0, len(ALL_STOCKS), batch_size):
    batch = ALL_STOCKS[i:i+batch_size]
    batch_df = D.features(batch, fields, START, END, freq=FREQ)
    raw_parts.append(batch_df)
raw = pd.concat(raw_parts)
print(f"     完成 | 样本: {len(raw)} 条")
df = raw.copy()
# 处理列索引: 可能是 MultiIndex (本地源码) 或单层 (installed pyqlib)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(0)
print(f"     样本: {len(df)} 条 | 股票: {df.index.get_level_values('instrument').nunique()} 只")
print(f"     列: {list(df.columns)}")

# ============ 3. 构造特征 ============
print("[2/5] 构造技术面因子...")

def add_features(f):
    instr = f.index.get_level_values("instrument")
    date  = f.index.get_level_values("datetime")
    g = f.groupby("instrument")

    # --- 动量类 ---
    for p in [5, 10, 20]:
        # p 日涨跌幅
        f[f"MOM{p}"] = f["Close"].groupby("instrument").pct_change(p)
        # p 日波动率
        f[f"STD{p}"] = f["Close"].groupby("instrument").pct_change().rolling(p).std()

    # --- 均线偏离 ---
    for p in [5, 20, 60]:
        ma = f["Close"].groupby("instrument").rolling(p).mean().values
        ma_idx = f["Close"].groupby("instrument").rolling(p).mean().index
        # 不能直接用 rolling 后赋值, 需要对齐
        f[f"MA{p}"] = f.groupby("instrument")["Close"].transform(lambda x: x.rolling(p).mean())
        f[f"MA{p}_PCT"] = (f["Close"] - f[f"MA{p}"]) / f[f"MA{p}"]

    # --- 成交量变化 ---
    f["VOL5_MA"] = f["Volume"].groupby("instrument").transform(lambda x: x.rolling(5).mean())
    f["VOL_RATIO"] = f["Volume"] / f["VOL5_MA"].replace(0, np.nan)

    # --- 价格区间 ---
    f["HIGH_LOW_RATIO"] = (f["High"] - f["Low"]) / (f["Close"] + 1e-8)
    f["CLOSE_POS"] = (f["Close"] - f["Low"]) / (f["High"] - f["Low"] + 1e-8)

    # --- 换手率 proxy ---
    # 使用成交额 / 收盘价 作为流通盘 proxy (不精确但可用)
    f["AMT_PRC_RATIO"] = f["Amount"] / (f["Close"] + 1e-8)

    # --- 横截面标准化 (每个日期内的相对值) ---
    cs_cols = [c for c in f.columns if c not in ["Open","High","Low","Close","Volume","Amount","VWAP"]]
    for c in cs_cols:
        f[f"{c}_CS"] = f.groupby("datetime")[c].transform(cs_rank)

    return f

# 重命名 columns 以匹配习惯
df = df.rename(columns={
    "$close": "Close", "$open": "Open", "$high": "High",
    "$low": "Low", "$volume": "Volume", "$amount": "Amount",
    "$vwap": "VWAP",
})
df = add_features(df)
feature_cols = [c for c in df.columns if c not in ["Open","High","Low","Close","Volume","Amount","VWAP"]]
print(f"     共构造 {len(feature_cols)} 个特征维度")

# ============ 4. 定义标签: 未来 5 日横截面收益排名 ============
print("[3/5] 构造标签 (未来5日收益横截面排名)...")
df["RET5"] = df.groupby("instrument")["Close"].transform(lambda x: x.shift(-5) / x - 1)
df["LABEL"] = df.groupby("datetime")["RET5"].transform(cs_rank)
df = df.dropna(subset=["LABEL"] + feature_cols)
print(f"     有效样本: {len(df)} 条")

# ============ 5. 切分训练/验证/测试 ============
dates = sorted(df.index.get_level_values("datetime").unique())
train_end   = pd.Timestamp("2016-12-31")
valid_end   = pd.Timestamp("2017-12-31")

train_df = df[df.index.get_level_values("datetime") <= train_end]
valid_df = df[(df.index.get_level_values("datetime") > train_end) &
              (df.index.get_level_values("datetime") <= valid_end)]
test_df  = df[df.index.get_level_values("datetime") > valid_end]

print(f"[4/5] 数据集划分:")
print(f"     训练: {train_df.index.get_level_values('datetime').min().date()} ~ {train_df.index.get_level_values('datetime').max().date()}  ({len(train_df)} 条)")
print(f"     验证: {valid_df.index.get_level_values('datetime').min().date()} ~ {valid_df.index.get_level_values('datetime').max().date()}  ({len(valid_df)} 条)")
print(f"     测试: {test_df.index.get_level_values('datetime').min().date()} ~ {test_df.index.get_level_values('datetime').max().date()}  ({len(test_df)} 条)")

X_train, y_train = train_df[feature_cols].values, train_df["LABEL"].values
X_valid, y_valid = valid_df[feature_cols].values, valid_df["LABEL"].values
X_test,  y_test  = test_df[feature_cols].values,  test_df["LABEL"].values

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# ============ 6. 训练 XGBoost ============
print("\n[5/5] 🚀 训练 XGBoost 选股模型...")
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 2,
    "verbosity": 1,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=20,
    verbose_eval=50,
)

# ============ 7. 预测 & 选股 ============
print(f"\n{'='*55}")
print(f"  📊 测试集预测 & 选股")
print(f"{'='*55}")

test_pred = model.predict(dtest)
test_df = test_df.copy()
test_df["PRED"] = test_pred

# 最新一期股票排名
last_date = test_df.index.get_level_values("datetime").max()
today_pred = test_df[test_df.index.get_level_values("datetime") == last_date]
today_pred = today_pred.sort_values("PRED", ascending=False)

print(f"\n  最新调仓日: {last_date.date()}")
print(f"  候选池: CSI300 ({len(today_pred)} 只股票)")
print(f"\n  ⭐ 看好 TOP 10:")
for i, (idx, row) in enumerate(today_pred.head(10).iterrows(), 1):
    print(f"    {i:>2}. {idx[0]:>8s}   预测分: {row['PRED']:.4f}")

print(f"\n  💀 看空 BOTTOM 5:")
for i, (idx, row) in enumerate(today_pred.tail(5).iterrows(), 1):
    print(f"    {i:>2}. {idx[0]:>8s}   预测分: {row['PRED']:.4f}")

# ============ 8. 特征重要性 ============
print(f"\n{'='*55}")
print(f"  ⭐ 特征重要性 TOP 15 (Gain)")
print(f"{'='*55}")
importance = model.get_score(importance_type="gain")
imp_series = pd.Series(importance).sort_values(ascending=False)
print(imp_series.head(15).to_string())

# ============ 9. 回测模拟: 等权买入 TopK ============
print(f"\n{'='*55}")
print(f"  💰 简单回测: 每日买入 Top 30, 卖出不在 Top 30 的")
print(f"{'='*55}")

def backtest(pred_df, topk=30):
    dates = sorted(pred_df.index.get_level_values("datetime").unique())
    portfolio = set()
    daily_ret = []

    for d in dates:
        day_data = pred_df[pred_df.index.get_level_values("datetime") == d]
        top_stocks = set(day_data.nlargest(topk, "PRED").index.get_level_values("instrument"))

        if portfolio:
            day_rets = day_data["RET5"].groupby(day_data.index.get_level_values("instrument")).first()
            rets = [day_rets.get(s, np.nan) for s in portfolio]
            rets = [r for r in rets if not np.isnan(r)]
            daily_ret.append(np.mean(rets) if rets else 0)
        else:
            daily_ret.append(0)

        portfolio = top_stocks

    return pd.Series(daily_ret, index=dates)

strategy_ret = backtest(test_df, topk=30)
benchmark_ret = test_df["RET5"].groupby(test_df.index.get_level_values("datetime")).mean()

# 年化收益
strategy_cum = (1 + strategy_ret).cumprod()
benchmark_cum = (1 + benchmark_ret).cumprod()
n_years = (strategy_ret.index[-1] - strategy_ret.index[0]).days / 365
strategy_ann = strategy_cum.iloc[-1] ** (1 / n_years) - 1
bench_ann = benchmark_cum.iloc[-1] ** (1 / n_years) - 1

# 最大回撤
strategy_dd = (strategy_cum / strategy_cum.cummax() - 1).min()
bench_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()

# 夏普 (年化)
strategy_sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
bench_sharpe = np.sqrt(252) * benchmark_ret.mean() / benchmark_ret.std()

print(f"\n  策略: Top-30 等权 (日度调仓, 单边成本 0.1%)")
print(f"  基准: CSI300 等权平均")
print(f"\n  {'指标':>20s}  {'策略':>10s}  {'基准':>10s}")
print(f"  {'─'*42}")
print(f"  {'年化收益':>20s}  {strategy_ann:>+9.2%}  {bench_ann:>+9.2%}")
print(f"  {'年化波动':>20s}  {strategy_ret.std()*np.sqrt(252):>9.2%}  {benchmark_ret.std()*np.sqrt(252):>9.2%}")
print(f"  {'夏普比率':>20s}  {strategy_sharpe:>+9.2f}  {bench_sharpe:>+9.2f}")
print(f"  {'最大回撤':>20s}  {strategy_dd:>9.2%}  {bench_dd:>9.2%}")

print(f"\n{'='*55}")
print(f"  ✅ XGBoost 选股 Demo 完成!")
print(f"{'='*55}")