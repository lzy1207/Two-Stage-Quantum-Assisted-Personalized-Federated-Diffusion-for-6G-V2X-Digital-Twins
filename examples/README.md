# 合成数据运行示例

`demo_run/` 是执行 `python -m qv2x demo --output examples/demo_run` 得到的真实产物，包括小规模训练权重、预测、日志及图。

配置：12张16×16合成地图、4个客户端、3轮联邦训练、24次扩散更新、8步DDIM。量子模块保留8 qubit、4层、728共享参数，采用解析期望/自动微分。

该演示只用于验证软件连通性，未充分训练；不能代替RadioMapSeer实验，Stage-II误差也不保证低于Stage-I。具体实测结果在 `demo_run/metrics.json`，没有填入论文表格数值。

`new_tx_query.npz` 只含一个Tx坐标与空白建筑语义图，不含真实信号，配套 `new_tx_prediction.npz` 是以下命令的结果：

```bash
python -m qv2x infer --run examples/demo_run --query examples/new_tx_query.npz --output examples/new_tx_prediction.npz --seed 42
```

示例checkpoint对应小型配置。正式配置需要重新训练；不要把示例权重加载进paper模型。
