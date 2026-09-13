# 因子 IC / ICIR 分析 Demo

用 Qlib 数据 API 对 A 股因子进行 **IC (Information Coefficient)** 和 **ICIR** 有效性评估的示例。

## 文件说明

| 文件 | 说明 |
|---|---|
| `factor_ic_analysis.py` | 主脚本：因子加载 → IC/ICIR 计算 → 分组收益 → 状态判定 |

## 运行方法

```bash
pip install pyqlib scipy

# 下载 A 股数据（如尚未下载）
python -c "
import qlib; qlib.init()
from qlib.tests.data import GetData
GetData().qlib_data(target_dir='~/.qlib/qlib_data/cn_data', region='cn')
"

# 运行
cd /csai-workspace/qlib
python Reference/demo_factor_ic/factor_ic_analysis.py
```

## 什么是 IC 和 ICIR？

| 指标 | 公式 | 含义 |
|---|---|---|
| **Rank IC** | Spearman(因子值, 下期收益) | 因子预测选股能力的**方向**和**强度** |
| **IC Mean** | mean(IC 序列) | 因子长期平均预测能力 |
| **ICIR** | mean(IC) / std(IC) | 因子预测能力的**稳定性**（信噪比） |
| **IC 胜率** | P(IC > 0) | 因子正向预测的胜率 |

### 判断标准

| ICIR | IC Mean | 判定 | 含义 |
|---|---|---|---|
| > 0.5 | > 0.03 | 🟢 强势因子 | 预测能力强且稳定 |
| 0.3 ~ 0.5 | > 0.02 | 🟡 观察 | 可用但不稳定 |
| < 0.2 | ~ 0 | 🔴 失效 | 已无预测能力 |

## 本 Demo 测试的因子

| 因子 | 表达式 | 逻辑 |
|---|---|---|
| ROC20 | 20日涨跌幅 | 中期反转 |
| ROC5 | 5日涨跌幅 | 短期反转 |
| VOLATILITY20 | 20日波动率 | 低波异常 |
| VOLUME_RATIO | 当日量/20日均量 | 放量信号 |
| HIGH_LOW | (高-低)/收盘 | 日内振幅 |
| AMOUNT_MA5 | 当日额/5日均额 | 资金活跃度 |