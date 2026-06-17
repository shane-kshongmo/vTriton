# profile_utilization.py 执行说明

本文档只说明 `perfbound/analyze/profile_utilization.py` 的执行命令、参数和输出文件。

## 基本命令

在仓库根目录执行：

```bash
python -m perfbound.analyze.profile_utilization \
  --op-summary path/to/op_summary.csv \
  --des-graph path/to/des.json \
  --calibration path/to/calib.json \
  --kernel-name kernel_name \
  --output-file path/to/profile_utilization_report.json
```

## 必需参数

| 参数 | 含义 |
| --- | --- |
| `--op-summary` | 输入的 op_summary CSV 文件路径 |
| `--des-graph` | 输入的 DES graph JSON 文件路径 |

## 可选参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--calibration` | `perfbound/calibration/data/calib_910b3_v1.json` | calibration JSON 文件路径 |
| `--kernel-name` | 不指定 | 从 `op_summary` 中选择要分析的算子名；不指定时读取第一行可用算子 |
| `--u-threshold` | `0.80` | utilization 判断阈值 |
| `--r-threshold` | `0.50` | residency 判断阈值 |
| `--work-tolerance` | `0.10` | work mismatch warning 的相对误差阈值 |
| `--t-bound-us` | 不指定 | 外部传入的 tight bound，单位 us |
| `--output-file` | `data/profile_utilization_inputs/profile_utilization_report.json` | profile utilization JSON 报告输出路径 |

## 输出文件

执行后得到一个文件：

| 文件 | 生成条件 |
| --- | --- |
| `profile_utilization_report.json` | 始终生成，路径由 `--output-file` 指定；如果不指定，则输出到 `data/profile_utilization_inputs/profile_utilization_report.json` |

命令行也会打印一份文本摘要。

## 示例

```bash
python -m perfbound.analyze.profile_utilization \
  --op-summary data/profile_utilization_inputs/op_summary.csv \
  --des-graph data/profile_utilization_inputs/des.json \
  --output-file data/profile_utilization_inputs/profile_utilization_report.json
```

执行后会生成：

```text
data/profile_utilization_inputs/profile_utilization_report.json
```
