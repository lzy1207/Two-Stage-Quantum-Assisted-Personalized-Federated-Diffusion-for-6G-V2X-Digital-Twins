# 实际验证报告

日期：2026-09-26。以下记录本次实际执行的验证，与论文原始实验结果区分。

## 环境

Windows，Python 3.11.9；PyTorch 2.5.1+cu121；NumPy 1.23.5；SciPy 1.10.1；Matplotlib 3.7.2；Pillow 10.4.0。CPU完整演示和单元测试使用2个计算线程；另做了CUDA前向、反向和采样检查。

## 自动测试

```bash
python -m unittest discover -s tests -v
```

**55项全部通过**。完整日志见 `test-results.txt`。包括：

- 量子门的解析结果、statevector范数、728参数、finite-shots统计、parameter-shift与自动微分/有限差分对照，包含两个编码门共享角的梯度。
- Top-K、量化误差、序列化往返、bitmap字节对齐、Table III四种payload数量。
- 数据格式、realization隔离、测试目标不影响训练样本、单图采样内存行为、manifest显式dB转换与Tx元数据。
- FL样本数加权、私有head保存、head与共享上传参数区分、置信度下溢防护、原型总方差。
- FL断点续训与连续训练逐参数一致；Stage-II续训与连续训练逐参数一致（此CPU配置）。
- diffusion正反向公式、最后一步无噪声、FiLM梯度、随机seed、SSIM范围、掩膜与物理单位指标。
- 完整流程、保存完整配置、无需maps_db的新Tx推理、缓存被修改时拒绝误用、稳定性统计。

## 实际运行

1. `python -m qv2x demo --output examples/demo_run`：完整Stage-I→条件构造→Stage-II→验证选择→测试评估。运行输出和小型训练权重随包交付。
2. `python -m qv2x infer --run examples/demo_run --query examples/new_tx_query.npz --output examples/new_tx_prediction.npz --seed 42`：查询文件没有目标信号，输出有限16×16预测地图。
3. `python -m qv2x stability --run examples/demo_run --samples 100 --output examples/demo_run/stability`：同一测试Tx、固定条件、100个不同噪声seed，保存CDF统计、逐样本RMSE和剖面。
4. 两个重建基线 `classical`、`fedavg_classical` 各运行2个seed的小型sweep，成功生成逐次CSV及stage分离的mean/std汇总。该检查使用2 qubit/1 layer、6张8×8合成图、1轮FL与2次diffusion更新，只检验实验调度；日志位于工作目录的 `runs/sweep_check`，不作为科研结果打包。
5. 使用正式量子参数（8 qubit、4层、100 shots、parameter-shift）对2个样本实际前向/反向，728个共享参数及全部梯度有限。
6. 使用正式U-Net宽度64/128/256/256、64维embedding，对batch=1、256×256输入实际前向，输出形状正确且有限；网络参数量17,602,753（1语义通道）。
7. CUDA上运行小型100-shots量子parameter-shift，以及diffusion训练损失反向与完整DDPM采样，梯度和输出有限。

## 合成示例的测量值

| 项目 | 实测 |
|---|---:|
| Stage-I测试coarse RMSE | 7.41125 dB |
| Stage-II测试RMSE | 9.67765 dB |
| Stage-II测试SSIM | 0.07548 |
| 100样本：固定test realization 7的单次RMSE均值 | 9.04221 dB |
| 100样本：单次RMSE标准差 | 0.40598 dB |

这些值来自24次扩散更新的未充分训练模型，**不能用来验证论文优势**。此例Stage-II误差高于Stage-I；保留真实结果，未选择更好seed、未按真值挑图、未将100个样本平均图的RMSE替代单次生成评分。100样本的统计对象是一个固定Tx，因此与整个测试集的一次采样RMSE不同。

## 未完成的科研验证范围

没有原始RadioMapSeer数据/划分文件/原始权重，未执行90客户端×400轮、20,000次diffusion、5个seed的正式训练；未验证论文0.89 dB结果、量子优势或全部外部算法基线。六维特征、测量算子、归一化和部分结构细节仍是明示的重建选择。CPU/CUDA可执行性不表示整个论文训练能在任意硬件上快速完成。

## 归档校验

发布脚本只包含项目代码、配置、文档、测试及明确标记的合成示例，排除临时runs和缓存。最终ZIP使用CRC检查，并包含覆盖所有交付文件的 `MANIFEST.sha256`。
