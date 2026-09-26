# IOTJ Quantum V2X — 完整流程重建代码

依据用户提供的 **IOTJ__Quantum_V2X.pdf（15页新版）**，参考 FL_v2x.rar 和 con_diffusion_v2x.zip，重新实现两阶段量子辅助个性化联邦扩散 REM 方法。

这是可训练、可推理、可测试的**重建实现**。原始训练数据、模型权重、随机划分及部分实现细节未包含在附件中，因此不是丢失代码的逐行恢复，也没有把论文表格里的数值当作本程序的测试结果。`configs/paper.json` 对齐论文明确列出的超参数；`configs/demo.json` 用小型合成数据检查软件全流程。正式 RadioMapSeer 实验需要自行接入数据并重新训练。

## 1. 安装与快速验证

建议 Python 3.10–3.12。CPU 可完成演示；正式 256×256 实验建议使用 CUDA GPU。无需 OpenAI API、云服务或真实量子设备。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m qv2x demo
```

Windows 也可在已安装依赖的环境运行 `./run_demo.ps1`。命令从解压后的项目根目录执行。演示保留真正的 **8 qubit / 4 layer / 728 个共享参数**量子电路，缩小客户端数量、网格、训练步数与 U-Net；采用精确期望值和解析自动微分加速。演示模型并未充分训练，其精度没有论文复现意义。

## 2. 已实现的内容

| 模块 | 实现 |
|---|---|
| 数据 | realization 级 70/10/20 划分，10×9 空间客户端，10% 邻区采样，训练集归一化，真实数据 manifest/NPZ 接口 |
| 混合量子骨干 | π·tanh 输入编码，RY/RZ 编码，RX/RY/RZ 可训练层，环形 CNOT，16个 Pauli 可观测量，32维输出 |
| 量子训练 | 可微 statevector；有限 shots 联合采样；parameter-shift 与 adjoint 两种梯度；共享输入角的两个编码门分别求导 |
| 个性化 FL | 32→16→1 本地 head，按样本数加权聚合，只上传共享参数，保存每个客户端 head、归一化和历史空间支持 |
| 通信 | Top-K bitmap、8/32 bit、可序列化报文，分别统计论文 payload 与包含 metadata 的实际 wire 大小；可选 error feedback |
| Stage-I 条件 | 历史覆盖和验证误差置信度；冻结网格查询生成粗 REM；量子均值和包含客户端间项的总方差 prototype |
| Stage II | 时间条件 U-Net，粗 REM/Tx Gaussian/语义条件，量子 prototype FiLM；noise + reconstruction + gradient + SSIM 损失 |
| 采样 | 完整 DDPM；显式可选 DDIM 快速采样；相同坐标与条件产生可控随机样本 |
| 实验 | RMSE、SSIM、Radial MAE、Trend Violation、ECDF、预测图、训练日志、通信曲线、多个 seeds/消融 |
| 工程 | CPU/CUDA、阶段化命令、Stage-I/II checkpoint、恢复训练、本地无标签在线推理、数学正确性与端到端测试 |

实现细节与公式对应关系见 [docs/PAPER_IMPLEMENTATION_MAP.md](docs/PAPER_IMPLEMENTATION_MAP.md)，源代码问题审计见 [docs/REFERENCE_AUDIT.md](docs/REFERENCE_AUDIT.md)。附带的 `IOTJ__Quantum_V2X.zip` 是较旧论文源码，其标题、数值及 prototype 公式与 PDF 不一致，未用旧稿覆盖新版定义。

## 3. 真实数据训练

数据入口是 `[M,H,W]` 的 dB 信号图、`[M,2]` 的归一化 Tx 坐标、可选语义图与六维环境特征。**Tx 坐标必须来自元数据；不从待预测信号图的最大值推断。建筑掩码来自语义地图；不从目标信号强弱推断。**

[RadioMapSeer 官方入口](https://radiomapseer.github.io/)及 [RadioUNet 作者代码](https://github.com/RonLevie/RadioUNet)可获取原始数据与格式说明。本项目不捆绑该数据集。不同发布版本/预处理存在差异，PNG→dB 映射必须由实际数据说明确认，转换器不会猜测阈值。详见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

```bash
# manifest 每行明确指定一个地图、Tx 坐标和可选语义文件
python -m qv2x import-data --manifest data/manifest.csv --output data/radiomapseer.npz
python -m qv2x run --config configs/paper.json --output runs/paper
```

正式配置按 Table I 使用 400 轮 FL、100 shots、parameter-shift、20,000 次 diffusion 更新、1,000 步 DDPM。该设置需要大量计算，尤其参数位移需要重复执行电路。可把 `quantum.gradient_method` 改成 `adjoint` 进行模拟器开发，但应在实验记录中说明此变化；有限 shots 前向加解析期望梯度并非真实硬件的完全随机梯度估计。`shots=0` 表示解析期望，不是100次测量。

论文未明确的六个环境特征、16个测量算子、U-Net 层宽、样本数、target normalization、radial bin数及 optimizer细节均已明确记录为重建假设。默认六特征为建筑占用、Tx距离、横纵坐标差、接收位置横纵正弦特征；有原始六维特征时直接传入 `features` 覆盖。

## 4. 分阶段运行与恢复

```bash
python -m qv2x stage1 --config configs/paper.json --output runs/paper
python -m qv2x conditions --config configs/paper.json --output runs/paper
python -m qv2x stage2 --config configs/paper.json --output runs/paper
python -m qv2x stage2 --config configs/paper.json --output runs/paper --resume
python -m qv2x evaluate --config configs/paper.json --output runs/paper
```

Stage-I 中断恢复：配置中设置 `federated.resume_from` 为之前的 `runs/paper/latest.pt`，将 `rounds` 设为期望总轮数，再运行 `stage1`。恢复时保留原数据、seed、client partition和模型结构。局部 optimizer 按轮重建，这是明确的实现选择。Stage-II 使用 `stage2_last.pt` 恢复优化器、迭代数与随机状态；验证集 RMSE 选择 `stage2_best.pt`，测试集从不参与 checkpoint 选择。

默认 FL 是单进程多客户端模拟；私有 head 在模拟器 checkpoint 中分别保存，真实部署应放在各客户端。代码没有提供车辆网络通信层，也不把“未上传原始数据”说成差分隐私保证。

## 5. 新 Tx 在线推理

准备查询 NPZ：`tx_xy [M,2]`，以及与训练一致的 `semantics [M,C,H,W]` 和/或 `features [M,6,H,W]`。完全无语义时提供 `grid_shape=[H,W]`。不需要 `maps_db` 或目标 Tx 的实测信号。

```bash
python -m qv2x infer --run runs/paper --query data/new_tx.npz --output runs/new_tx_prediction.npz --seed 42
```

实现使用同一冻结 Stage-I 模型、历史 spatial support 和 validation reliability，同时生成 coarse REM 与 prototype，再运行 diffusion。重复使用相同 seed 可重现同一模拟器采样路径。

## 6. 消融与多次实验

```bash
python -m qv2x sweep --config configs/paper.json --output runs/ablation --seeds 0 1 2 3 4 --variants full classical no_entanglement no_personalization no_confidence no_coarse no_heatmap no_prototype no_structural_losses
python -m qv2x sweep --config configs/paper.json --output runs/compression --seeds 0 1 2 3 4 --variants no_compression quantization_only topk_only full
```

`fedavg_classical` 是全局经典模型的 FedAvg 实现；`classical` 是本项目中 728 参数匹配的经典替代模块；`coordinate_ddpm` 将 coarse/prototype/semantic 条件置零，仅保留 Tx heatmap，是匹配网络结构的坐标条件消融。`no_structural_losses` 仅保留噪声预测损失。上述名称均不冒充他人发布的完整算法。附件没有提供 FedASA、WFL-VTopK、pFedWN、RadioUNet、Classical-LDM 的完整实现和训练结果，因此本包**不声称已经重现所有外部论文基线**；可把其预测结果接入 `qv2x.metrics.evaluate_maps` 进行统一比较。论文中本方法的主流程及消融开关均可运行。

固定 `split_seed` 保持不同方法一致划分；合成数据的生成 seed 默认为 `split_seed`，可用 `data.seed` 单独指定，因此 sweep 的训练 seed 不会更换合成地图。正式公平比较应使用同一真实 NPZ。`summary.json` 汇总各 stage 的 mean/std，`runs.csv` 保存逐次结果。不会按 ground truth 挑选“最佳”生成图。

同一测试 Tx 条件下的重复生成、CDF均值/标准差及横纵剖面：

```bash
python -m qv2x stability --run runs/paper --samples 100 --output runs/paper/stability
```

更多实验设置见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 7. 输出与指标

`runs/<name>/` 包括完整配置、环境、数据 fingerprint、split、Stage-I 全部客户端状态、Stage-II 最优与最新模型、条件 NPZ、真实推理预测、JSON/CSV日志和 PNG 图。样例输出见 `examples/demo_run/`；所有示例都明确标记为 synthetic。

RMSE 按所有有效像素 SSE 汇总后开方；SSIM 按每张图计算后平均，采用训练地图有效像素范围确定固定 dynamic range；训练采用 z-score，SSIM loss 的范围同步除以训练标量标准差。`mean_power_db` 是dB值均值，`mean_power` 是使用训练集 min/max 仿射归一化后的均值（不裁剪）。径向指标仅统计非空 bins，无相邻有效 bin 时 trend 为 `null`，有效区域由外部 mask 给定。online coarse/prototype 通信开销与 FL 参数更新分开；论文 Table III payload不包括量化scale、报头及online融合数据，本代码额外报告 wire 大小。默认3轮demo日志中的 FL RMSE 是客户端验证测量 RMSE，最终 `metrics.json` 的 Stage-I 是完整粗地图 RMSE。

量子统计、压缩数量、量子梯度、条件无标签依赖及短程训练通过测试不代表量子优势得到验证。正式论文数字需要原始数据版本、准确特征和足量训练后重新测量。

## 8. 项目结构

```text
qv2x/           数据、量子电路、联邦学习、压缩、条件扩散、指标和命令行
configs/        论文配置与快速演示配置
tests/          unittest 自动验证
docs/           公式对应、来源审计、数据格式和测试报告
examples/       实际运行的合成示例和可视化
scripts/        辅助脚本
```

源附件中的代码只作为设计参考，重建模块不要求运行 Colab notebook、不绑定私人 Google Drive 路径。依赖版本范围为可安装范围；本次实际验证版本记录在 `docs/VALIDATION.md`。
