# 数据格式与真实数据导入

真实实验数据通过 CSV 清单导入为统一的 NPZ。清单明确给出每个 realization 的无线电图、发射位置和可选环境信息。程序不会从无线电图的最亮位置推断发射坐标，也不会根据目标功率推断建筑。

RadioMapSeer 数据与原始格式请参见 [RadioMapSeer 项目主页](https://radiomapseer.github.io) 和 [RadioUNet 官方代码库](https://github.com/RonLevie/RadioUNet)。本工程不附带完整数据集，也不自动下载。不同数据版本的 gain、path loss、功率和灰度编码可能不同，导入前需要依据所用版本的说明确认单位、坐标及编码。

## CSV 清单

CSV 使用 UTF-8，可带 BOM。相对路径以 **CSV 所在目录** 为基准；也支持绝对路径。包含逗号的路径要使用标准 CSV 双引号。

```csv
realization_id,map_path,tx_x,tx_y,building_path,road_path
scene001_tx00,radio/scene001_tx00.npy,0.25,0.75,geometry/scene001_buildings.png,geometry/scene001_roads.png
scene001_tx01,radio/scene001_tx01.npy,0.80,0.30,geometry/scene001_buildings.png,geometry/scene001_roads.png
scene002_tx00,radio/scene002_tx00.npy,0.40,0.60,geometry/scene002_buildings.png,
```

| 字段 | 含义 |
|---|---|
| `realization_id` | 必填，唯一非空字符串。一个 ID 对应一个 Tx 与场景配置。它仅保存在元数据中，不进入模型输入。 |
| `map_path` | 必填，二维 `.npy` 射频数值图，或需要显式校准的 8-bit 灰度 `.png`。 |
| `tx_x`, `tx_y` | 必填，发射机 x/y 坐标。x 沿图像列方向，y 沿行方向。默认已归一化到 `[0,1]`。 |
| `building_path` | 可选，独立建筑占据图，`.npy` 或 `.png`；零为空地，正值为建筑。 |
| `road_path` | 可选，独立道路占据图，`.npy` 或 `.png`；零为非道路，正值为道路。 |
| `features_path` | 可选，含六个环境描述通道的 `.npy`，形状 `[6,H,W]`。 |
| `semantics_path` | 可选，自定义空间语义 `.npy`，形状 `[C,H,W]`。与 `building_path`/`road_path` 二选一。 |

同一清单的所有无线电图必须使用同一个 `(H,W)`。所有环境图必须与对应无线电图像素对齐，导入器不自动裁剪、拉伸、插值或更改坐标。`features_path`、`semantics_path` 一旦使用，必须为每一行提供，且语义通道数 C 必须一致。建筑/道路路径可以逐行留空，空值按全零占据图处理。

默认输入坐标是归一化坐标。若 CSV 写的是像素坐标，使用 `--coordinates pixel`，导入会计算 `tx_x/(W-1)`、`tx_y/(H-1)`；允许边界坐标，拒绝越界值。只有一个像素的轴对应坐标必须为零。

## 射频数值、单位与 PNG 校准

### 已有真实射频数值的 NPY

`.npy` 必须是二维实数数值数组。导入器认为其中已经是所声明类型的 dB/dBm 数值，**不会自动把 `[0,1]` 或 `0..255` 数组转为 dB**。从第三方生成的归一化 NPY 导入前，应先按其原始定义还原射频数值。

接收功率 dBm 数据示例：

```bash
python -m qv2x import-data --manifest data/manifest.csv --output data/real_maps.npz --value-kind received_power
```

三种输入语义：

| `--value-kind` | 输入含义 | 导入后的 `maps_db` |
|---|---|---|
| `received_power` | 接收功率 dBm | 原值保持不变；`tx_power_dbm` 不改变此类输入。 |
| `gain` | 信道增益 dB | 未给 Tx 功率时保留 dB 增益；给出 `--tx-power-dbm P` 时变为 `P + gain_db`，单位 dBm。 |
| `pathloss` | 路径损耗 dB | 必须给出 `--tx-power-dbm P`，变为 `P - pathloss_db`，单位 dBm。 |

例如，若输入 NPY **已经确认**为正值路径损耗、对应参考发射功率为 23 dBm：

```bash
python -m qv2x import-data --manifest data/pathloss_manifest.csv --output data/received_power.npz --value-kind pathloss --tx-power-dbm 23 --coordinates pixel
```

此转换只包含给出的标量发射功率。如果原数据还需要天线增益、参考电平或其他校准项，应在导入前按数据定义计算，不要把它们默认为零后声称恢复了原实验。

### PNG 显式校准

无线电 PNG 只接受 Pillow 模式 `L` 的 **8-bit 单通道灰度图**。RGB 彩色图、伪彩色 heatmap、调色板图和 16-bit PNG 会被拒绝，因为没有通用且可靠的隐式功率映射。

必须同时提供 `--png-db-min` 和 `--png-db-max`，它们表示像素值 0 和 255 分别对应的射频值：

```text
value_db = pixel / 255 * (png_db_max - png_db_min) + png_db_min
```

两端必须是有限值，并满足 `png_db_min < png_db_max`。下面用变量占位，**没有替您假设 RadioMapSeer 的实际标定上下限**：

```powershell
# 从所用数据版本的说明中读取并填写这两个值。
$dbAtPixelZero = <confirmed_value_at_0>
$dbAtPixel255 = <confirmed_value_at_255>
python -m qv2x import-data --manifest data/png_manifest.csv --output data/real_maps.npz --png-db-min $dbAtPixelZero --png-db-max $dbAtPixel255 --value-kind gain
```

上面的 `<...>` 是需要替换的说明占位符，不是可以直接执行的 PowerShell 数值。若数据采用反向灰度、非线性压缩、截断或特殊无效值编码，请先依据官方定义解码成真正的 `.npy` 射频数组，再导入；此处的线性正向映射不适用于这些情况。PNG 的零值不会自动被当作建筑或无效像素。

## 环境图、特征与有效像素

建筑/道路图来自独立几何数据。占据图 `.npy` 可为 bool 或有限非负数，PNG 可为二值模式 `1` 或灰度模式 `L`。导入保持像素格，统一以 `value>0` 得到 0/1 占据图。使用这些路径时，语义通道顺序固定为：

```text
semantics[:,0] = building_occupancy
semantics[:,1] = road_occupancy
```

显式建筑掩膜中的建筑像素不计入有效目标像素；道路图不会额外改变有效像素。没有建筑路径时不会用功率值猜测建筑。无线电 NPY 中非有限值也被标为无效，存储时替换为有限占位值以避免数值传播。每个 realization 至少需要一个有效像素。

`semantics_path` 是一般语义数组，不自动转换为二值，也不会自动把任一通道解释为评估掩膜。此模式下有效像素来自射频值的有限性。如果需要精确的自定义有效掩膜，可直接构造下面的规范 NPZ，而不是让导入器猜测语义通道的含义。

使用 `features_path` 时，提供的六个环境通道原样进入模型输入。它们必须在推理时可取得，不能包含目标接收功率、目标图统计量、目标误差、realization ID 或测试集标签。

不提供六通道文件时，`data.get_features` 使用明确记录的重建默认值：语义通道 0（无则为零）、Tx/Rx 归一化距离除以 sqrt(2)、横向绝对距离、纵向绝对距离、`sin(pi*rx_x)`、`sin(pi*rx_y)`。它们只依赖坐标与已知语义；不是从丢失源码恢复出的原始六特征。最终逐点输入始终为 `[tx_x,tx_y,rx_x,rx_y,六个描述量]` 共 10 维。

## 规范 NPZ

导入结果只包含数值数组和 JSON 字符串，使用 `allow_pickle=False` 加载：

当前加载器将 NPZ 数组整体读入 RAM；压缩 NPZ 不提供真正的 mmap 访问。导入也会把每张 map/语义累积后堆叠。完整大型数据集可能显著超过压缩文件大小，导入前应估计 `M×H×W×通道数×4 bytes` 及中间副本、mask 和训练条件的额外内存。根据硬件选择明确的子集，或先扩展磁盘分片/mmap 接口；客户端特征抽样按单图处理并不消除这些全量数组的内存需求。

| 数组 | 形状与含义 |
|---|---|
| `maps_db` | float32 `[M,H,W]`，声明的 dB/dBm 射频值。 |
| `tx_xy` | float32 `[M,2]`，归一化 Tx `(x,y)`。 |
| `valid_mask` | bool `[M,H,W]`，有效目标/评估像素。 |
| `semantics` | 可选 float32 `[M,C,H,W]`，已知环境通道。 |
| `features` | 可选 float32 `[M,6,H,W]`，已知环境特征。 |
| `metadata_json` | 标量 UTF-8 JSON 字符串，记录 ID、来源类型、校准与单位。 |

也可直接用 Python 构造 NPZ；这适用于更复杂的原始标定和有效像素规则：

```python
import numpy as np
from qv2x.data import MapDataset, save_dataset

dataset = MapDataset(
    maps_db=np.load("decoded_power_dbm.npy", allow_pickle=False),  # [M,H,W]
    tx_xy=np.load("tx_normalized_xy.npy", allow_pickle=False),    # [M,2]
    valid_mask=np.load("valid_pixels.npy", allow_pickle=False),  # [M,H,W]
    semantics=np.load("known_semantics.npy", allow_pickle=False),
    metadata={"signal_unit": "dBm", "source": "explicitly_decoded_real_data"},
)
save_dataset(dataset, "data/real_maps.npz")
```

射频目标只能用于训练监督、验证及最终测试。训练/验证/测试按整个 transmitter-conditioned realization 分开，不能随机把同一张图的像素分到不同集合。若要求未知城市场景泛化，还应使用比论文 realization 划分更严格的场景级分组划分。测试集上的特征提取与 coarse/prototype 构造必须只读取 Tx、接收网格、已知语义及冻结模型，不能读取目标功率。

这些导入检查证明数据接口和单位声明得到执行，不能替代对原始数据含义的核实，也不能保证达到论文中的数值结果。
