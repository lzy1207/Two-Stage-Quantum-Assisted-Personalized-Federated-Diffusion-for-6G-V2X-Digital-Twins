# 训练、推理与实验协议

本工程提供论文主方法和可追溯的重建消融。**论文规模的 RadioMapSeer 训练尚未运行，文档中的命令是运行方法，不是已取得论文数值的证明。** 已执行的测试和合成演示以 `VALIDATION.md` 及其保存的真实日志为准。

所有命令从项目根目录运行。配置是 JSON，路径按当前工作目录解释。对算法、数据、seed 或量子结构的变更使用新的输出目录，避免把旧条件缓存或旧模型混入新实验。先通过 `python -m qv2x --help` 查看当前可用子命令。

## 快速软件检查

```bash
python -m unittest discover -s tests -v
python -m qv2x demo --config configs/demo.json --output runs/demo
python -m qv2x plot --run runs/demo
```

`demo` 与 `run` 执行相同的完整流程，只是默认配置不同。默认演示为 12 张 16×16 人工传播图、2×2 客户端网格、3 轮 FL、24 次扩散训练更新。量子部分仍为 8 qubit、4 layer、728 个共享参数；使用 `shots=0` 和 `adjoint`，并以 8 步 DDIM 采样 32 步扩散模型。它验证接口与数值通路，不能用来论证拟合精度或量子优势。

随机性分为三层：顶层 `seed=42` 控制默认演示模型/训练及客户端抽样；`split_seed=2025` 固定 realization 划分；合成数据使用 `data.seed`（如显式配置），否则使用 `split_seed`。因此默认合成数据 seed 是 2025，不随 sweep 的模型 seed 改变。

## 真实数据全流程

先按 `DATA_FORMAT.md` 准备真实射频数值、独立 Tx 元数据和环境图，再导入：

```bash
python -m qv2x import-data --manifest data/manifest.csv --output data/radiomapseer.npz
python -m qv2x run --config configs/paper.json --output runs/paper --device auto
```

上面的导入命令适用于 `.npy` 已为接收功率 dBm 的情况；PNG、gain、pathloss 必须按实际单位补充相应参数。`paper.json` 默认读取 `data/radiomapseer.npz`。模型不会自动下载数据，也不会自动猜测 PNG 的物理标定。

`run` 顺序执行 Stage I → 冻结模型生成条件 → Stage II → held-out 测试及绘图。正式配置对齐 PDF Table I：400 轮、2 个 local epochs、100 shots、parameter-shift；扩散 20,000 次更新、1,000 步 DDPM。其余明确超参数见 `PAPER_IMPLEMENTATION_MAP.md`。完整论文实验还需要真实 256×256 地图、正确的特征定义和五个独立 seed；单条 `run` 只运行一个 seed。

重建配置中的每客户端最多 1,024 个训练观测、Adam/AdamW、U-Net 内部结构、特征默认配方及归一化不是原稿完整指定的信息。需要替换成已找回的原始实验设置时，记录具体改动。只有实际拥有训练和验证样本的客户端才参与，运行日志中的客户端数是通信成本的真实依据。

当前规范 NPZ 是**内存中读取**：全部原始 maps、semantics、可选 features 与最后堆叠的 coarse 条件缓存会占用 RAM。客户端特征抽样已按单张图处理，以减少额外临时特征内存，但并没有把整套数据访问变成磁盘流式训练。压缩包体积也不等于解压后的 float32 数组体积。处理完整 RadioMapSeer 前，先根据 RAM/显存预处理选择明确记录的子集；需要更大规模时，应增加磁盘分片或 mmap 数据接口。不要把合成演示成功解释为任意笔记本都能加载全部数据。256×256 的完整 U-Net、batch size 16 和全量量子训练同样需要单独评估算力与显存预算。

重建验证已覆盖默认 8-qubit/4-layer/100-shot parameter-shift 的小批量反向传播，以及通道宽度 64/128/256/256 的完整 U-Net 在 256×256、batch size 1 下的前向计算。它们验证相关代码路径，不能代替 batch size 16 的长期稳定训练或 400 轮/20,000 更新的论文实验。

## 分阶段训练与恢复

```bash
python -m qv2x stage1 --config configs/paper.json --output runs/paper
python -m qv2x conditions --config configs/paper.json --output runs/paper
python -m qv2x stage2 --config configs/paper.json --output runs/paper
python -m qv2x evaluate --config configs/paper.json --output runs/paper
```

| 命令 | 主要输入 | 主要产物 |
|---|---|---|
| `stage1` | 数据与客户端划分 | `stage1.pt`、`latest.pt`、联邦训练日志、每客户端 head/历史接收位置/验证可靠性 |
| `conditions` | 当前 `stage1.pt`、已知坐标/环境 | `conditions.npz` 的 coarse REM 与 prototype，以及 `conditions_manifest.json` |
| `stage2` | 相同数据与已验证条件缓存 | `stage2_last.pt`、验证 RMSE 选择的 `stage2_best.pt`、`diffusion_history.json` |
| `evaluate` | Stage-II 最优 checkpoint、测试 split | `metrics.json`、测试预测、图像 |
| `plot` | 已保存的预测和日志 | 重新绘制对比图、曲线；不会重新训练 |

条件缓存记录 Stage-I checkpoint、数据和 split 的 SHA-256。改变 Stage I 后需要重新构建条件并重新训练 Stage II；程序会拒绝使用不匹配的缓存和 checkpoint。所有条件计算使用冻结模型与已知环境，目标图只用于监督和评估。

Stage-I 续训：复制原 JSON，设置 `federated.resume_from` 为当前 `latest.pt`，把 `federated.rounds` 改为**期望总轮数**，然后运行 `stage1`。数据、seed、client partition、模型结构与个性化方式必须保持一致。私有 head、共享模型、历史支持和随机状态由 checkpoint 恢复；本实现的本地 Adam 每轮重建，不恢复跨轮本地 optimizer 状态，因为原训练协议本来就不保留该状态。

Stage-II 续训：将原配置的 `diffusion.iterations` 改为**期望总更新数**，保持结构与上游条件一致，执行：

```bash
python -m qv2x stage2 --config configs/paper_extended.json --output runs/paper --resume
python -m qv2x evaluate --config configs/paper_extended.json --output runs/paper
```

`--resume` 只用于 `stage2`；恢复 `stage2_last.pt` 的模型、优化器、更新计数与随机状态。验证与保存发生在 `validate_every` 的间隔和最后一次更新；意外中断时从最近保存的更新恢复。测试集从不参与最优 checkpoint 选择。

## 新 Tx 的无标签在线推理

查询 NPZ 不包含目标射频图，包含：

- `tx_xy`：`[M,2]`，归一化坐标。
- 与训练一致的 `semantics [M,C,H,W]` 和/或 `features [M,6,H,W]`。
- 两者都不提供时，用 `grid_shape=[H,W]` 给出接收网格。

语义通道数必须与训练模型一致。特征含义、坐标归一化与已知环境的预处理也应一致。

```bash
python -m qv2x infer --run runs/paper --query data/new_tx.npz --output runs/new_tx_prediction.npz --seed 42
```

输出包含 `prediction_db`、`coarse_db`、`prototype` 和 `tx_xy`。此命令没有目标图，不能自动计算 RMSE。相同 seed 在同一运行环境中控制量子 shots 和扩散随机路径；改变 seed 可以产生不同预测。推理采用训练目录保存的采样设置，而不是从新配置文件推断。

## 多 seed 与消融实验

```bash
python -m qv2x sweep --config configs/paper.json --output runs/ablation --seeds 0 1 2 3 4 --variants full classical no_entanglement no_personalization no_confidence no_coarse no_heatmap no_prototype no_structural_losses
```

每个 `variant/seed_N` 独立运行全流程。`runs.csv` 保留每个 Stage 的逐次指标；`summary.json` 对 RMSE、SSIM、Radial MAE 和 Trend Violation 分别报告均值和样本标准差（`ddof=1`；只有一次运行时标准差记为零）。固定 `split_seed`、同一真实 NPZ 和相同 seed 列表，才能进行配对比较。

| variant | 实际改变 |
|---|---|
| `full` | 当前配置原样运行。 |
| `classical` | 用参数匹配的经典替代 backbone 替换量子电路，默认共享参数仍为 728。 |
| `no_entanglement` | 移除环形 CNOT，保留全部旋转参数。 |
| `no_personalization` | 共享并聚合预测 head；通信参数量随之增加。 |
| `no_confidence` | 去掉置信度加权，改用均匀空间可靠性。 |
| `no_coarse` / `no_heatmap` / `no_prototype` | 对相应条件置零，其他训练流程保持。 |
| `no_structural_losses` | reconstruction、gradient 和 SSIM 权重全部置零，仅训练 noise loss。 |
| `fedavg_classical` | 本工程经典 backbone + 全局共享 head，dense FP32 上传。 |
| `coordinate_ddpm` | coarse、prototype 和语义条件置零，只保留 Tx 条件。 |

`classical`、`fedavg_classical` 和 `coordinate_ddpm` 是这里实际实现的对照定义，不是完整复刻外部发表的 Classical-LDM、FedASA、WFL-VTopK、pFedWN、RadioUNet 或 CCDDPM。若接入外部实现，要单独记录其来源、训练协议、参数量和预测文件，不能把占位结果写成论文 baseline 成绩。

## 量子 shots、深度和梯度方式

复制配置，为每组设置单独 JSON 和输出目录。以下字段控制实验：

| 字段 | 示例取值 | 解释 |
|---|---|---|
| `quantum.shots` | `0`, `10`, `100`, `1000` | 0 为精确期望；其他值为联合测量采样次数。比较时明确区分。 |
| `quantum.n_layers` | `1`, `2`, `4`, `6` | PQC 深度。改变参数量、运行成本及上传负载。 |
| `quantum.entanglement` | `true`, `false` | 保留/关闭 ring CNOT。 |
| `quantum.gradient_method` | `parameter_shift`, `adjoint` | 严格偏移电路采样，或模拟器精确期望梯度。 |
| `query_batch_size` | 如 `256` 或 `1024` | 网格查询批大小，影响内存和速度，不是模型表达能力参数。 |

例如，用 Python 生成一组有清晰来源的配置再运行：

```python
import json
from pathlib import Path

config = json.loads(Path("configs/paper.json").read_text(encoding="utf-8"))
config["quantum"]["shots"] = 1000
Path("configs/shots1000.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
```

```bash
python -m qv2x sweep --config configs/shots1000.json --output runs/shots1000 --variants full --seeds 0 1 2 3 4
```

8 qubit、4 layer 的 parameter-shift 反向传播，每批量梯度需要 `2×96 + 4×8 = 224` 次偏移电路计算，再加前向计算。100 shots 又增加测量抽样成本。该实现按经典 statevector 模拟，不能据此声称实际量子硬件加速。

使用 `adjoint` 且 shots>0 时，前向是有限采样，反向是精确期望梯度的替代估计；不能将其结果标为完整的 finite-shot parameter-shift 实验。深度 L 改变时，默认其他维度不变，共享参数量为 `632 + 24L`，所以通信开销也必须重新计算。

## 结构损失与采样实验

`diffusion.loss_weights` 顺序为 `[lambda_rec,lambda_grad,lambda_ssim]`，默认 `[1.0,0.1,0.1]`；noise loss 的权重固定为 1。分别移除单项时可设置 `[0,0.1,0.1]`、`[1,0,0.1]` 或 `[1,0.1,0]`。`no_structural_losses` 会同时关闭三项，而不只是梯度和 SSIM。

`diffusion.mask_training_loss=false` 为当前默认，整图计算训练损失；设置 true 是额外的 mask-aware 训练方案，应记录为协议变化。它不会把推理条件改为需要目标掩膜。

`diffusion.sample_steps=null` 或等于 `diffusion.timesteps` 使用完整 ancestral DDPM；小于总步数使用 eta=0 DDIM。后者只是一种明确的加速采样设置，不能称为论文的完整 1,000 步 reverse chain。模型训练时间步数 `timesteps` 与实际生成步数 `sample_steps` 是不同参数。修改采样设置时使用新实验配置和目录，保存 sampler 名称与步数以便比较。

## 压缩实验与开销统计

```bash
python -m qv2x sweep --config configs/paper.json --output runs/compression --seeds 0 1 2 3 4 --variants no_compression quantization_only topk_only full
```

| variant | Top-K ratio | bits |
|---|---:|---:|
| `no_compression` | 1.0 | 32 |
| `quantization_only` | 1.0 | 8 |
| `topk_only` | 0.5 | 32 |
| `full`（以 paper 配置为前提） | 0.5 | 8 |

也可自行设置 `federated.topk_ratio` 与 `quant_bits`（只支持 8 或 32）。默认不使用 `federated.error_feedback`；启用它属于额外算法变体。`paper_bits` 按论文计算保留值和 bitmap，不含 metadata；`wire_bits` 是本实现序列化消息长度，包含 10-byte header 和字节对齐。它仍不是包含车辆网络协议、重传和下行的实际无线链路总开销。

只有 728 个上传参数、90 个参与客户端、400 轮以及默认个性化方式同时成立时，四组论文 payload 才分别为 104.832、26.208、55.692、16.380 MB。非个性化变体还上传 head；深度或客户端数改变时不能直接套用这些总数。

## 固定条件下的生成稳定性

```bash
python -m qv2x stability --run runs/paper --samples 100 --seed 7000
python -m qv2x stability --run runs/paper --realization-id 12 --samples 100 --output runs/paper/stability_tx12 --seed 7000
```

`--realization-id` 是规范 NPZ 中的整数地图下标，不是 CSV 的字符串 ID，必须属于保存的 test split；省略时选第一个 test realization。该命令读取测试真值用于离线评估，与不需要真值的 `infer` 用途不同。

固定同一缓存的 coarse REM 和 prototype，生成 100 个独立扩散样本，统计逐像素均值/标准差、ECDF 均值/标准差和经过 Tx 的横纵剖面。它不重新抽取量子 shots，因此衡量的是 **固定 Stage-I 条件后的扩散采样稳定性**，而不是整个系统所有随机因素。所有样本均纳入，不按真值挑选最好样本。

输出目录默认是 `runs/paper/stability_tx_<id>/`：`sample_stats.npz`、`stability_summary.json`、`cdf_stability.png`、`profile_stability.png`。NPZ 包含 mean/std map、逐次 RMSE、ECDF 统计、采样 seeds、Tx、真值与 mask。summary 中 `rmse_mean_db` 是逐样本 RMSE 的平均，`rmse_std_db` 是其样本标准差；`mean_map_rmse_db` 则是所有样本平均图的 RMSE，两者含义不同，不能替换论文的单次生成误差。

## 指标、单位和结果阅读

- `rmse`：对所有有效像素的 squared error 汇总后开方，单位 dB；分别列出 Stage-I coarse 与 Stage-II final。
- `ssim`：直接在物理 dB 图像上计算完整图的局部 SSIM，再作均值；固定 `data_range_db` 默认为完整训练图有效像素的 max-min，也可由 `evaluation.data_range_db` 显式指定。预测与真值不会各自归一化。
- 训练中的 SSIM loss 在 Stage-I scaler 的 z-score 空间计算，range 为 `data_range_db/scaler.std`。它不等于评估物理 dB SSIM，因为共同平移会影响 SSIM 的亮度项。z-score 没有固定 `[-1,1]` 上下界。
- `radial_mae`：比较 Tx 周围相同径向 bin 的预测/真值平均 dB；空 bin 跳过。`trend_violation`：相邻且均非空 bins 的向外正增长均值；没有可用相邻 bins 时为 null。
- `mean_power_db`：有效像素上预测 dB 的算术均值，不是线性功率平均。
- `mean_power`：用完整训练图有效像素的 min/range 归一化预测后求均值，不 clip 到 `[0,1]`。这是本项目对论文未明确定义的归一化作出的选择，不保证等于论文 Table V。

训练 target scaler 只从 Stage-I 的稀疏训练测量拟合；评估范围使用完整训练图，不使用 val/test。所有尺度保存在运行记录/模型中。比较不同方法时，要固定数据、划分、尺度、mask、径向分箱和采样协议；不要仅因为都输出 “SSIM” 就认为定义完全一致。

正式结果报告应同时保存 config、环境、数据 fingerprint、split、训练日志、checkpoint、预测和指标。测试通过与短程演示说明实现可以运行且满足所检查的数学性质；论文的最终精度、泛化与量子优势仍需真实数据和充分训练独立验证。
