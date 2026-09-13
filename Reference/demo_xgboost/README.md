# XGBoost 股票选择 Demo

使用 XGBoost 对 A 股（CSI300 成分股）进行收益预测和选股的完整示例。

## 文件说明

| 文件 | 说明 |
|---|---|
| `xgboost_stock_selector.py` | 主脚本：数据加载 → 因子构造 → XGBoost 训练 → 选股 → 回测 |

## 运行方法

```bash
pip install pyqlib xgboost

# 下载 A 股数据到 ~/.qlib/qlib_data/cn_data/（如尚未下载）
python -c "
import qlib; qlib.init()
from qlib.tests.data import GetData
GetData().qlib_data(target_dir='~/.qlib/qlib_data/cn_data', region='cn')
"

# 运行 Demo
cd /csai-workspace/qlib
python Reference/demo_xgboost/xgboost_stock_selector.py
```

## 流程概览

```
CSI300 日频行情 → 34 个技术面因子 → 横截面 Rank 标签 → XGBoost 训练 → 选股推荐
```

- **数据**: CSI300 成分股日频行情 (2015-2020)
- **特征**: 动量(5/10/20日)、波动率、均线偏离、成交量比、价格位置等
- **标签**: 未来 5 日收益的横截面排名（Rank）
- **模型**: XGBoost 回归 (early stopping)
- **选股**: 按预测得分排序，推荐 Top 10 / Bottom 5
- **回测**: Top-30 等权策略，对比 CSI300 等权基准