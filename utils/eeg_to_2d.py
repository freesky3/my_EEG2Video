# input data file: data\PSD_DE\watching
# 再上面的文件夹中，每个数据的形状都是(2, 5, 50, 62, 5)
# 我希望重新插值排布为(2, 5, 50, 5, 32, 32)

import mne
import numpy as np
from scipy.interpolate import griddata
import os
import glob
from tqdm import tqdm

# 标准10-20电极布局的62个电极名称
CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
    'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

def get_eeg_sensor_positions():
    """
    获取EEG电极的2D位置坐标，基于标准10-20电极布局。
    使用完整的自定义位置映射，确保所有62个电极都有位置。
    
    Returns:
        dict: 电极名称到(x, y)坐标的映射
    """
    # 完整的62个电极位置映射，基于标准10-20系统的扩展
    # 坐标系统：x轴从左到右(-1到1)，y轴从后到前(-1到1)
    positions = {
        # 前额区域 (Frontal-Polar)
        'FP1': [-0.3, 0.9],    # 左前额
        'FPZ': [0.0, 0.9],     # 前额中央
        'FP2': [0.3, 0.9],     # 右前额
        
        # 前额-额叶区域 (Anterior Frontal)
        'AF3': [-0.2, 0.7],    # 左前额-额叶
        'AF4': [0.2, 0.7],     # 右前额-额叶
        
        # 额叶区域 (Frontal)
        'F7': [-0.6, 0.5],     # 左额叶外侧
        'F5': [-0.4, 0.5],     # 左额叶中侧
        'F3': [-0.2, 0.5],     # 左额叶
        'F1': [-0.1, 0.5],     # 左额叶内侧
        'FZ': [0.0, 0.5],      # 额叶中央
        'F2': [0.1, 0.5],      # 右额叶内侧
        'F4': [0.2, 0.5],      # 右额叶
        'F6': [0.4, 0.5],      # 右额叶中侧
        'F8': [0.6, 0.5],      # 右额叶外侧
        
        # 额叶-中央区域 (Frontal-Central)
        'FT7': [-0.7, 0.3],    # 左额叶-颞叶
        'FC5': [-0.4, 0.3],    # 左额叶-中央外侧
        'FC3': [-0.2, 0.3],    # 左额叶-中央
        'FC1': [-0.1, 0.3],    # 左额叶-中央内侧
        'FCZ': [0.0, 0.3],     # 额叶-中央中央
        'FC2': [0.1, 0.3],     # 右额叶-中央内侧
        'FC4': [0.2, 0.3],     # 右额叶-中央
        'FC6': [0.4, 0.3],     # 右额叶-中央外侧
        'FT8': [0.7, 0.3],     # 右额叶-颞叶
        
        # 颞叶区域 (Temporal)
        'T7': [-0.8, 0.1],     # 左颞叶
        'T8': [0.8, 0.1],      # 右颞叶
        
        # 中央区域 (Central)
        'C5': [-0.4, 0.1],     # 左中央外侧
        'C3': [-0.2, 0.1],     # 左中央
        'C1': [-0.1, 0.1],     # 左中央内侧
        'CZ': [0.0, 0.1],      # 中央中央
        'C2': [0.1, 0.1],      # 右中央内侧
        'C4': [0.2, 0.1],      # 右中央
        'C6': [0.4, 0.1],      # 右中央外侧
        
        # 颞叶-顶叶区域 (Temporal-Parietal)
        'TP7': [-0.7, -0.1],   # 左颞叶-顶叶
        'TP8': [0.7, -0.1],    # 右颞叶-顶叶
        
        # 中央-顶叶区域 (Central-Parietal)
        'CP5': [-0.4, -0.1],   # 左中央-顶叶外侧
        'CP3': [-0.2, -0.1],   # 左中央-顶叶
        'CP1': [-0.1, -0.1],   # 左中央-顶叶内侧
        'CPZ': [0.0, -0.1],    # 中央-顶叶中央
        'CP2': [0.1, -0.1],    # 右中央-顶叶内侧
        'CP4': [0.2, -0.1],    # 右中央-顶叶
        'CP6': [0.4, -0.1],    # 右中央-顶叶外侧
        
        # 顶叶区域 (Parietal)
        'P7': [-0.6, -0.3],    # 左顶叶外侧
        'P5': [-0.4, -0.3],    # 左顶叶中侧
        'P3': [-0.2, -0.3],    # 左顶叶
        'P1': [-0.1, -0.3],    # 左顶叶内侧
        'PZ': [0.0, -0.3],     # 顶叶中央
        'P2': [0.1, -0.3],     # 右顶叶内侧
        'P4': [0.2, -0.3],     # 右顶叶
        'P6': [0.4, -0.3],     # 右顶叶中侧
        'P8': [0.6, -0.3],     # 右顶叶外侧
        
        # 顶叶-枕叶区域 (Parietal-Occipital)
        'PO7': [-0.5, -0.5],   # 左顶叶-枕叶外侧
        'PO5': [-0.3, -0.5],   # 左顶叶-枕叶中侧
        'PO3': [-0.2, -0.5],   # 左顶叶-枕叶
        'POZ': [0.0, -0.5],    # 顶叶-枕叶中央
        'PO4': [0.2, -0.5],    # 右顶叶-枕叶
        'PO6': [0.3, -0.5],    # 右顶叶-枕叶中侧
        'PO8': [0.5, -0.5],    # 右顶叶-枕叶外侧
        
        # 小脑区域 (Cerebellum)
        'CB1': [-0.3, -0.8],   # 左小脑
        'CB2': [0.3, -0.8],    # 右小脑
        
        # 枕叶区域 (Occipital)
        'O1': [-0.2, -0.7],    # 左枕叶
        'OZ': [0.0, -0.7],     # 枕叶中央
        'O2': [0.2, -0.7],     # 右枕叶
    }
    
    print(f"成功获取 {len(positions)} 个电极的位置")
    
    # 验证所有电极都有位置
    missing_channels = [ch for ch in CHANNEL_NAMES if ch not in positions]
    if missing_channels:
        print(f"警告: 仍有缺失的电极位置: {missing_channels}")
    else:
        print("✅ 所有62个电极位置都获取成功！")
    
    return positions

def eeg_data_to_2d_images(raw_data, channel_names, grid_res=32, method='cubic', 
                         sensor_positions=None, grid_x=None, grid_y=None):
    """
    将EEG数据转换为2D地形图像序列。
    
    Args:
        raw_data (np.ndarray): EEG数据数组 (n_channels, n_samples)
        channel_names (list): 对应raw_data的通道名称列表
        grid_res (int): 输出网格的分辨率 (grid_res x grid_res)
        method (str): 插值方法 ('linear', 'nearest', 'cubic')
        sensor_positions (dict, optional): 预计算的电极位置
        grid_x (np.ndarray, optional): 预计算的X网格
        grid_y (np.ndarray, optional): 预计算的Y网格
    
    Returns:
        np.ndarray: 2D图像张量 (n_samples, grid_res, grid_res)
    """
    # 如果没有提供预计算的电极位置，则计算一次
    if sensor_positions is None:
        sensor_positions = get_eeg_sensor_positions()
    
    # 提取我们需要的电极位置
    points = np.array([sensor_positions[ch] for ch in channel_names])
    
    # 如果没有提供预计算的网格，则计算一次
    if grid_x is None or grid_y is None:
        # 创建目标2D网格
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # 扩展边界以确保所有电极都在网格内
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        grid_x, grid_y = np.mgrid[
            (x_min - x_padding):(x_max + x_padding):complex(grid_res),
            (y_min - y_padding):(y_max + y_padding):complex(grid_res)
        ]
    
    # 对每个时间样本进行插值
    n_samples = raw_data.shape[1]
    images = np.zeros((n_samples, grid_res, grid_res))
    
    for i in tqdm(range(n_samples), desc="Converting to 2D images"):
        # 获取当前时间样本的数据
        values = raw_data[:, i]
        # 将值插值到网格上
        images[i] = griddata(points, values, (grid_x, grid_y), method=method, fill_value=0)
    
    return images

def reshape_data_for_cnn(data, grid_res=32):
    """
    将形状为(2, 5, 50, 62, 5)的数据重新排列为(2, 5, 50, 5, 32, 32)
    
    Args:
        data (np.ndarray): 输入数据，形状为(2, 5, 50, 62, 5)
        grid_res (int): 2D图像的分辨率
    
    Returns:
        np.ndarray: 重新排列后的数据，形状为(2, 5, 50, 5, 32, 32)
    """
    print(f"原始数据形状: {data.shape}")
    
    # 获取原始形状
    n_subjects, n_conditions, n_trials, n_channels, n_features = data.shape
    
    # 预计算电极位置和插值网格（只计算一次）
    print("预计算电极位置和插值网格...")
    sensor_positions = get_eeg_sensor_positions()
    points = np.array([sensor_positions[ch] for ch in CHANNEL_NAMES])
    
    # 创建目标2D网格（只计算一次）
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # 扩展边界以确保所有电极都在网格内
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    grid_x, grid_y = np.mgrid[
        (x_min - x_padding):(x_max + x_padding):complex(grid_res),
        (y_min - y_padding):(y_max + y_padding):complex(grid_res)
    ]
    
    # 正确的数据重塑：先转置，确保电极维度在最后
    # 从 (2, 5, 50, 62, 5) 转置为 (2, 5, 50, 5, 62)
    # 然后重塑为 (2*5*50*5, 62)
    print("重新排列数据维度...")
    data_transposed = data.transpose(0, 1, 2, 4, 3)  # (2, 5, 50, 5, 62)
    reshaped_data = data_transposed.reshape(-1, n_channels)  # (2*5*50*5, 62)
    
    print(f"重塑后数据形状: {reshaped_data.shape}")
    
    # 转换为2D图像，使用预计算的电极位置和网格
    print("转换为2D图像...")
    images_2d = eeg_data_to_2d_images(
        reshaped_data.T, CHANNEL_NAMES, grid_res=grid_res,
        sensor_positions=sensor_positions, grid_x=grid_x, grid_y=grid_y
    )
    
    print(f"2D图像形状: {images_2d.shape}")
    
    # 重新排列回原始结构，但将62个通道替换为32x32图像
    # 从(2*5*50*5, 32, 32)重塑为(2, 5, 50, 5, 32, 32)
    final_shape = (n_subjects, n_conditions, n_trials, n_features, grid_res, grid_res)
    result = images_2d.reshape(final_shape)
    
    print(f"最终数据形状: {result.shape}")
    
    return result

def process_data_files(input_dir, output_dir, grid_res=32):
    """
    处理指定目录中的所有数据文件。
    
    Args:
        input_dir (str): 输入数据目录路径
        output_dir (str): 输出数据目录路径
        grid_res (int): 2D图像的分辨率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.npy文件
    data_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not data_files:
        print(f"在 {input_dir} 中没有找到.npy文件")
        return
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    for file_path in tqdm(data_files, desc="处理数据文件"):
        # 加载数据
        data = np.load(file_path)
        
        # 检查数据形状
        if data.shape != (2, 5, 50, 62, 5):
            print(f"警告: 文件 {file_path} 的形状为 {data.shape}，期望为 (2, 5, 50, 62, 5)")
            continue
        
        # 转换为2D
        data_2d = reshape_data_for_cnn(data, grid_res=grid_res)
        
        # 保存结果
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, data_2d)
        
        print(f"已保存: {output_path}")

def visualize_eeg_layout():
    """
    可视化EEG电极布局，用于验证电极位置。
    """
    import matplotlib.pyplot as plt
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    sensor_positions = get_eeg_sensor_positions()
    
    # 绘制电极位置
    plt.figure(figsize=(12, 10))
    
    for ch_name, (x, y) in sensor_positions.items():
        plt.scatter(x, y, s=100, alpha=0.7)
        plt.annotate(ch_name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('EEG Electrode Layout (Standard 10-20 System)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = "data/PSD_DE/watching" # imaging, watching
    output_directory = "data/PSD_DE/watching_2d" # imaging_2d, watching_2d
    
    # 处理数据文件
    process_data_files(input_directory, output_directory, grid_res=32)
    
    # 可选：可视化电极布局
    # visualize_eeg_layout()
    
    print("EEG到2D转换完成！")