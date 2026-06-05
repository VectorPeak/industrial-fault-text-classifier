# Step 5 Evaluation And Inference 技术说明

## 0x01. 目标

在测试集上评估模型泛化能力，并提供单条报修文本推理入口。

## 0x02. 输入与输出

| 项目 | 路径示例 | 说明 |
|------|----------|------|
| 模型目录 | `artifacts/models/baseline` | Step 4 输出 |
| 测试集 | `data/processed/splits/test.csv` | Step 3 输出 |
| 评估报告 | `artifacts/reports/eval_report.json` | 三任务指标和混淆统计 |
| 预测结果 | `artifacts/reports/predictions.csv` | 测试集预测标签 |

## 0x03. 评估指标

| 指标 | 说明 |
|------|------|
| accuracy | 单任务准确率 |
| macro-F1 | 不受类别频次主导的整体分类质量 |
| P0/P1 recall | 高风险工单召回能力 |
| three-task exact match | 三个任务同时预测正确的比例 |
| confusion top | 高频误分组合，用于定位标签边界 |

工业报修场景中，高风险召回和处理部门误分往往比整体准确率更影响业务价值。

## 0x04. 运行命令

当前统一评估与单条推理入口面向 `naive_bayes` 后端，用于公开样例和本地数据的快速闭环验证。BERT 后端训练入口已保留，但尚未接入本步骤的统一评估与推理接口。

测试集评估：

```powershell
python scripts\step5_evaluate_and_predict.py evaluate --model-dir artifacts\models\baseline --data data\processed\splits\test.csv --report artifacts\reports\eval_report.json --predictions artifacts\reports\predictions.csv
```

单条文本预测：

```powershell
python scripts\step5_evaluate_and_predict.py predict --model-dir artifacts\models\baseline --text "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
```

## 0x05. 注意事项

当前公开全量 CSV 为基于 Kaggle 公开企业数据集清洗增强后的实验数据，用于复现流程与验证建模方案，不代表真实企业工单上线表现。正式评估应使用经过授权、脱敏和人工复核的数据，并保留独立测试集。
