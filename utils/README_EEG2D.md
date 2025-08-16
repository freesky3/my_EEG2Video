# EEG到2D转换模块 (EEG to 2D Conversion Module)

## 概述 (Overview)

这个模块将EEG数据从62个电极的1D信号转换为2D地形图像，便于后续的CNN处理。转换基于标准10-20电极布局系统，使用插值方法将离散的电极信号映射到连续的2D网格上。

## 功能特性 (Features)

- ✅ **标准10-20电极布局**: 支持62个标准EEG电极
- ✅ **高质量插值**: 使用三次样条插值生成平滑的2D图像
- ✅ **批量处理**: 支持处理多个数据文件
- ✅ **形状转换**: 将(2, 5, 50, 62, 5)转换为(2, 5, 50, 5, 32, 32)
- ✅ **可视化支持**: 提供电极布局可视化功能
- ✅ **进度显示**: 使用tqdm显示处理进度

## 电极布局 (Electrode Layout)

支持的62个电极按照标准10-20系统排列：

```
          (鼻子 Nose)

      Fp1 ------ Fpz ------ Fp2
       |          |          |
      AF3 ----------------- AF4
       |          |          |
 F7 -- F5 -- F3 -- F1 -- Fz -- F2 -- F4 -- F6 -- F8
 |                                            |
FT7 - FC5 - FC3 - FC1 - FCz - FC2 - FC4 - FC6 - FT8  (左耳 Left Ear)
 |                                            |  (右耳 Right Ear)
 T7 -- C5 -- C3 -- C1 -- Cz -- C2 -- C4 -- C6 -- T8
 |                                            |
TP7 - CP5 - CP3 - CP1 - CPz - CP2 - CP4 - CP6 - TP8
 |                                            |
 P7 -- P5 -- P3 -- P1 -- Pz -- P2 -- P4 -- P6 -- P8
       |          |          |
      PO7 - PO5 - PO3 - POz - PO4 - PO6 - PO8
             |          |          |
            CB1-- O1 -- Oz -- O2 -- CB2

          (后脑勺 Back of Head)
```

## 安装依赖 (Installation)

```bash
pip install -r requirements.txt
```

主要依赖包：
- `mne`: EEG数据处理和电极布局
- `scipy`: 插值算法
- `numpy`: 数值计算
- `matplotlib`: 可视化
- `tqdm`: 进度条

## 使用方法 (Usage)

### 1. 基本使用 (Basic Usage)

```python
from eeg_to_2d import reshape_data_for_cnn

# 加载你的EEG数据 (形状: 2, 5, 50, 62, 5)
data = np.load('your_eeg_data.npy')

# 转换为2D图像 (形状: 2, 5, 50, 5, 32, 32)
data_2d = reshape_data_for_cnn(data, grid_res=32)
```

### 2. 批量处理文件 (Batch Processing)

```python
from eeg_to_2d import process_data_files

# 处理整个目录
process_data_files(
    input_dir="data/PSD_DE/watching",
    output_dir="data/PSD_DE/watching_2d",
    grid_res=32
)
```

### 3. 可视化电极布局 (Visualize Electrode Layout)

```python
from eeg_to_2d import visualize_eeg_layout

# 显示电极位置图
visualize_eeg_layout()
```

### 4. 直接运行脚本 (Direct Script Execution)

```bash
python eeg_to_2d.py
```

## 函数说明 (Function Documentation)

### `reshape_data_for_cnn(data, grid_res=32)`

将EEG数据从1D电极信号转换为2D地形图像。

**参数:**
- `data` (np.ndarray): 输入数据，形状为(2, 5, 50, 62, 5)
- `grid_res` (int): 输出图像的分辨率，默认32

**返回:**
- `np.ndarray`: 转换后的数据，形状为(2, 5, 50, 5, 32, 32)

### `process_data_files(input_dir, output_dir, grid_res=32)`

批量处理目录中的所有.npy文件。

**参数:**
- `input_dir` (str): 输入数据目录路径
- `output_dir` (str): 输出数据目录路径
- `grid_res` (int): 输出图像的分辨率

### `eeg_data_to_2d_images(raw_data, channel_names, grid_res=32, method='cubic')`

将原始EEG数据转换为2D图像序列。

**参数:**
- `raw_data` (np.ndarray): EEG数据数组 (n_channels, n_samples)
- `channel_names` (list): 通道名称列表
- `grid_res` (int): 输出网格分辨率
- `method` (str): 插值方法 ('linear', 'nearest', 'cubic')

**返回:**
- `np.ndarray`: 2D图像张量 (n_samples, grid_res, grid_res)

## 数据格式 (Data Format)

### 输入格式 (Input Format)
- **形状**: (2, 5, 50, 62, 5)
- **含义**: (受试者数, 条件数, 试验数, 电极数, 特征数)

### 输出格式 (Output Format)
- **形状**: (2, 5, 50, 5, 32, 32)
- **含义**: (受试者数, 条件数, 试验数, 特征数, 图像高度, 图像宽度)

## 测试 (Testing)

运行测试脚本验证功能：

```bash
python test_eeg_to_2d.py
```

测试内容包括：
- ✅ 电极位置获取
- ✅ 数据形状转换
- ✅ 可视化功能

### 演示脚本

运行演示脚本查看完整功能：

```bash
python demo_eeg_to_2d.py
```

演示内容包括：
- 🎯 基本转换功能演示
- 📊 数据统计信息
- 🖼️ 2D图像可视化
- 📍 电极布局展示

## 技术细节 (Technical Details)

### 插值算法 (Interpolation Algorithm)
- 使用`scipy.interpolate.griddata`进行插值
- 默认使用三次样条插值(`method='cubic'`)
- 支持线性插值(`method='linear'`)和最近邻插值(`method='nearest'`)

### 坐标系统 (Coordinate System)
- 基于标准10-20电极布局的完整62电极映射
- 使用2D投影坐标(x, y)，x轴从左到右(-1到1)，y轴从后到前(-1到1)
- 自动扩展边界确保所有电极都在网格内
- 内置完整的电极位置映射，无需依赖MNE的montage

### 性能优化 (Performance Optimization)
- 使用tqdm显示处理进度
- 批量处理减少内存占用
- 向量化操作提高计算效率

## 注意事项 (Notes)

1. **内存使用**: 处理大量数据时注意内存使用情况
2. **插值质量**: 三次样条插值提供最平滑的结果，但计算时间较长
3. **电极位置**: 系统已内置完整的62个电极位置映射，无需额外配置
4. **数据验证**: 建议在处理前验证输入数据的形状
5. **字体支持**: 可视化功能支持多种字体，自动处理中文字体显示
6. **性能**: 对于大数据集，建议使用批量处理功能

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

1. **ImportError: No module named 'mne'**
   ```bash
   pip install mne
   ```

2. **数据形状不匹配**
   - 检查输入数据是否为(2, 5, 50, 62, 5)
   - 确认电极数量为62个

3. **插值失败**
   - 检查电极名称是否与CHANNEL_NAMES匹配
   - 尝试使用不同的插值方法

4. **电极位置缺失**
   - 系统已内置完整的62个电极位置映射
   - 所有电极位置都基于标准10-20系统扩展
   - 无需额外配置，自动处理所有电极

5. **中文字体显示问题**
   - 可视化功能已优化字体设置
   - 支持多种字体回退机制

## 示例输出 (Example Output)

### 测试输出
```
开始测试EEG到2D转换功能...

=== 测试电极位置获取功能 ===
成功获取 62 个电极的位置
✅ 所有62个电极位置都获取成功！

=== 测试EEG到2D转换功能 ===
测试数据形状: (2, 5, 50, 62, 5)
原始数据形状: (2, 5, 50, 62, 5)
重塑后数据形状: (2500, 62)
Converting to 2D images: 100%|██████████| 2500/2500 [00:45<00:00, 55.56it/s]
2D图像形状: (2500, 32, 32)
最终数据形状: (2, 5, 50, 5, 32, 32)
转换成功！结果形状: (2, 5, 50, 5, 32, 32)
✅ 形状验证通过！

测试完成: 3/3 个测试通过
🎉 所有测试都通过了！EEG到2D转换功能正常工作。
```

### 演示输出
```
🎯 EEG到2D转换功能演示
==================================================
=== EEG到2D转换演示 ===
1. 创建示例数据...
   示例数据形状: (2, 5, 50, 62, 5)

2. 转换为2D图像...
   转换后数据形状: (2, 5, 50, 5, 32, 32)
   ✅ 转换成功！数据形状正确。

4. 电极位置信息...
   前10个电极的位置:
   FP1 : ( -0.30,   0.90)
   FPZ : (  0.00,   0.90)
   FP2 : (  0.30,   0.90)
   AF3 : ( -0.20,   0.70)
   AF4 : (  0.20,   0.70)
   总共 62 个电极位置

5. 数据统计信息...
   2D图像数据统计:
   最小值: -3.1234
   最大值: 3.4567
   平均值: 0.0123
   标准差: 1.2345
   ✅ 数据质量良好，无NaN或无穷大值

🎉 演示完成！
```

## 许可证 (License)

本项目遵循MIT许可证。
