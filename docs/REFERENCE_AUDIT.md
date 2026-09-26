# 原始文件审计与重建边界

本项目是依据论文及两个参考代码包重新实现的可运行工程。它不是从训练结果逆向恢复出的原始源码，也不包含无法取得的原始数据集、模型权重和实验随机划分。原始文档和脚本中的操作说明仅作为参考材料，没有作为用户的新指令执行。

## 源文件身份

以下 SHA-256 在本次重建时对本地输入文件计算，可用于核对来源；文件内容未修改。

| 文件 | SHA-256 |
|---|---|
| `IOTJ__Quantum_V2X.pdf` | `3BBE7B6ED681FBA9FBEEFCA0CA5A53320BC9BCAC983EAE738903B8C6F79AE6A6` |
| `FL_v2x.rar` | `75DFFEC6E0C3CAA78F39210C71E1EFD52D03C45D899D68766F47FFCC4B6B0377` |
| `con_diffusion_v2x.zip` | `73CBF887F656F896E42CA6B4564C09802422EC79478938C581B9F770E66120DC` |
| 工作目录额外发现的旧稿 `IOTJ__Quantum_V2X.zip` | `856B0F4FA164E5EC42EFD9A026EB3AEB0FE1DAFAF910086CEB545A1F9896021F` |

优先级为：用户指定 PDF 中的公式、算法与 Table I > 两个参考工程的设计 > 旧版 LaTeX。PDF 标题为 *Two-Stage Quantum-Assisted Personalized Federated Diffusion for 6G V2X Digital Twins*，共 15 页。参数表与通信表另外渲染第 10 页进行了视觉核验。旧稿与 PDF 的结果数字、阶段定义和原型构造不同，不能混用。

## 从 FL 代码继承的设计

`FL_v2x.rar` 中的核心文件为 `1/datazhizuo.py`、`1/buzhou2.py`、`1/buzhou3.py`。

- 数据制作脚本采用空间网格划分 90 个客户端；原脚本把四个发射源的功率与坐标组成 `raw6=[x,y,sA,sB,sC,sD]`，还按信号离散程度划分异质性组。
- 训练脚本保留客户端个性化 head，聚合共享 extractor/backbone，支持稀疏上传、量化、通信日志和检查点。
- 推理脚本按坐标预测四个发射源的输出，依赖旧工程固定路径和全局统计量。

重建采用其个性化联邦训练与共享参数更新思想，同时改为 PDF 的 **Tx/Rx 坐标 + 6 个环境描述量 → 单个接收功率** 接口。原脚本中的四个信号值不能当作环境特征直接输给目标模型，否则会把待预测场信息混入条件。客户端邻域比例使用 PDF 的 10%；原参考脚本说明中的 20% 并非论文默认值。通信 MB 使用 PDF 的十进制 10^6 bytes，而不是原 FL 脚本的 1024^2。

## 从 diffusion 代码继承的设计

`con_diffusion_v2x.zip` 中的 `con_diffusion_v2x/Diffusion10.ipynb` 提供坐标条件 DDPM、U-Net、噪声 MSE、断点保存、生成样本与绘图设计。其 `UNet2DModel` 使用通道宽度 `(64,128,256,256)`、每层两个 residual block，以及部分分辨率的 attention；这些属于参考实现信息，PDF 没有规定完全相同的 U-Net 细节。

重建把原来的单个 Tx 条件扩充为 PDF 的 coarse REM、Tx 高斯热图、可选语义通道，并加入量子原型的 FiLM 调制和结构损失。纯 PyTorch U-Net 的具体 block 布局、激活及归一化是明确的工程重建选择，不能宣称是丢失模型的逐层恢复。

原 notebook 包含从目标图像寻找最亮点以推断 Tx，以及对多次生成样本按与真值的匹配程度挑选 Top-K 的分析代码。本项目的正式推理应从独立 Tx 元数据读取发射位置；评估应对预先确定的样本计算指标，不以目标误差筛选最好生成图。参考 notebook 中的这些做法不能直接转为无泄漏的论文评测协议。

## 量子部分：明确规格与重建选择

论文明确规定 8 qubits、4 层 RX/RY/RZ + CNOT ring、输入 `pi*tanh(Linear(10,8))`、16 维量子观测、32 维共享特征、32→16→1 的私有 head，以及默认 100 shots。共享参数严格为 728：输入层 88、PQC 96、输出层 544。head 参数不计入共享上传负载。

量子模块使用 PyTorch 复数 statevector 模拟真实门操作，并提供有限 shots、parameter-shift 和可微 statevector/adjoint 路径。这里的 “adjoint” 配置表示可微模拟器的精确反向传播接口，并不声称使用特定硬件或某个外部 SDK 的实现。`adjoint` 加有限 shots 时采用采样前向和精确期望梯度的替代估计；`parameter_shift` 则对偏移后的电路也按配置采样。这两种估计不能混称。执行方式要记录在配置和结果中。`shots=0` 的快速确定性检查用于验证公式；它不等同于论文默认 100-shot 实验。parameter-shift 检查尤其覆盖同一个输入角同时出现在 RY 与 RZ 两个门中的链式求导；默认电路每次梯度更新需要 224 次偏移后的批量电路计算，另加一次前向，计算成本明显增加。

PDF 未列出 16 个具体 observable。默认重建采用 8 个单比特 Z 与 8 个相邻 ring ZZ；这是可解释且可用同一组计算基测量估计的选择，但属于假设。PDF 也未列出六个环境描述量的精确含义；项目提供的几何/语义特征配方必须视为默认重建配方，可用用户原始特征替换。特征与语义只能来自已知 Tx/Rx 和环境信息，不能从目标 radio map 反推。

## 与旧 LaTeX 的关键差别

| 项目 | PDF，应采用 | 旧稿，不作为本次目标 |
|---|---|---|
| coarse 与 end-to-end 区分 | Stage I 1.29/.962；Stage II .89/.972 | 将 1.29/.962 写作 end-to-end |
| 量子原型 | 冻结共享 QNN 对目标 Tx 的完整接收网格查询，按历史覆盖权重融合 | 部分段落依赖目标 realization 的测量 |
| 全局方差 | 包含客户端内方差及客户端均值差平方 | 较早原型定义不同 |
| 通信统计 | 728 参数，104.832→16.380 MB，减少 84.4% | 另一组总量与 67.4% |

## 尚未提供的复现实验输入

- 实际 RadioMapSeer 文件、采用的仿真变体、城市/Tx 子集与原始 split manifest。
- PNG 灰度或原始数值到功率 dB 的准确映射、参考发射功率、有效像素掩膜细则。
- 原来的六个环境特征、16 个 observable、稀疏轨迹与采样密度。
- 原训练 checkpoint、原始 seed、优化器状态、完整 baseline 实现及结果日志。
- U-Net 的完整层级设置、各处激活、优化器与损失 reduction、SSIM 窗口和径向分箱数。

其中射频单位尤其需要保持明确：只有输入数据具有已确认的 dB 语义，且预测已反归一化至同一尺度时，误差才能标为 dB。对普通灰度图或归一化数组运行成功不能自动证明取得论文中的 dB 结果。合成示例使用声明的人工传播场验证流程，不是 RadioMapSeer 数据。

## 结果与 baseline 的诚信边界

本项目的 synthetic/demo 运行用于检查数据、训练、条件构造、生成、指标及 checkpoint 全链路。小规模配置、少量训练迭代、快速采样和精确期望值模式得到的结果均不能用来声称复现论文的 0.89 dB、SSIM 0.972 或量子性能优势。文档中的论文数值仅是来源中报告的参考值，不是重建工程的新测试成绩。

默认演示使用模型 seed 42，独立的数据与划分 seed 为 2025；`sweep` 不会随模型 seed 重建不同的默认合成地图。真实数据实验应继续固定同一 NPZ、划分 seed、客户端采样协议及指标定义。此次重建未运行论文规模的 400 轮、20,000 次更新 RadioMapSeer 实验。

指标实现还做了以下可追溯选择：训练目标使用训练稀疏测量拟合的 z-score，不是固定 `[-1,1]`；训练 SSIM 的 range 按 z-score 单位调整，评估 SSIM 则直接作用于物理 dB 图像。`mean_power_db` 是预测 dB 值的算术平均；单独输出的 `mean_power` 使用完整训练图有效像素的 min/range 做线性归一化且不截断。PDF 未指定归一化细则，因此后者不能被解释为已经复现论文 Table V。具体定义见 `PAPER_IMPLEMENTATION_MAP.md` 与 `EXPERIMENTS.md`。

论文比较的 FedASA、WFL-VTopK、pFedWN、RadioUNet、CCDDPM、Classical-LDM 并没有随文件提供完整且一致的实验实现。本项目若提供 classical backbone 或消融配置，只应按实际算法和开关命名；不能将简化代理命名成上述完整 baseline，不能把没有运行的数字写入结果表。

本文档配合 `PAPER_IMPLEMENTATION_MAP.md` 阅读。后者给出逐公式接口、参数表、通信算术与验证门槛，供继续补齐原始实验数据后进行论文结果复现实验。
