"""
EEG到2D转换功能演示脚本
演示如何使用eeg_to_2d模块将EEG数据转换为2D图像
"""

import numpy as np
import matplotlib.pyplot as plt
from eeg_to_2d import reshape_data_for_cnn, get_eeg_sensor_positions, visualize_eeg_layout

def demo_basic_conversion():
    """
    演示基本的EEG到2D转换功能
    """
    print("=== EEG到2D转换演示 ===")
    
    # 创建示例数据：形状为(2, 5, 50, 62, 5)
    # 模拟2个受试者，5个条件，50个试验，62个电极，5个特征
    print("1. 创建示例数据...")
    sample_data = np.load(r"data\PSD_DE\watching\chenjiaxin_20250630_session1__PSD_DE.npy")
    print(f"   示例数据形状: {sample_data.shape}")
    
    # 转换为2D图像
    print("\n2. 转换为2D图像...")
    data_2d = reshape_data_for_cnn(sample_data, grid_res=32)
    print(f"   转换后数据形状: {data_2d.shape}")
    
    # 验证转换结果
    expected_shape = (2, 5, 50, 5, 32, 32)
    if data_2d.shape == expected_shape:
        print("   ✅ 转换成功！数据形状正确。")
    else:
        print(f"   ❌ 转换失败！期望形状: {expected_shape}, 实际形状: {data_2d.shape}")
    
    return data_2d

def demo_visualize_sample_image(data_2d):
    """
    演示可视化转换后的2D图像
    """
    print("\n3. 可视化示例2D图像...")
    
    # 选择一个样本进行可视化
    # 选择：受试者0，条件0，试验0，特征0
    sample_image = data_2d[0, 0, 0, 0, :, :]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sample_image, cmap='viridis', aspect='equal')
    plt.colorbar(label='EEG Signal Value')
    plt.title('Sample EEG Topographic Map (2D Image)')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.tight_layout()
    plt.show()
    
    print("   ✅ 图像可视化完成！")

def demo_electrode_positions():
    """
    演示电极位置获取功能
    """
    print("\n4. 电极位置信息...")
    
    positions = get_eeg_sensor_positions()
    
    # 显示前10个电极的位置
    print("   前10个电极的位置:")
    for i, (ch_name, (x, y)) in enumerate(list(positions.items())[:10]):
        print(f"   {ch_name:4s}: ({x:6.2f}, {y:6.2f})")
    
    print(f"   总共 {len(positions)} 个电极位置")

def demo_data_statistics(data_2d):
    """
    演示数据统计信息
    """
    print("\n5. 数据统计信息...")
    
    print(f"   2D图像数据统计:")
    print(f"   最小值: {data_2d.min():.4f}")
    print(f"   最大值: {data_2d.max():.4f}")
    print(f"   平均值: {data_2d.mean():.4f}")
    print(f"   标准差: {data_2d.std():.4f}")
    
    # 检查是否有NaN或无穷大值
    has_nan = np.isnan(data_2d).any()
    has_inf = np.isinf(data_2d).any()
    
    if not has_nan and not has_inf:
        print("   ✅ 数据质量良好，无NaN或无穷大值")
    else:
        print("   ⚠️ 数据中存在NaN或无穷大值")

def main():
    """
    主演示函数
    """
    print("🎯 EEG到2D转换功能演示")
    print("=" * 50)
    
    try:
        # 1. 基本转换演示
        data_2d = demo_basic_conversion()
        
        # 2. 电极位置演示
        demo_electrode_positions()
        
        # 3. 数据统计演示
        demo_data_statistics(data_2d)
        
        # 4. 可视化演示
        demo_visualize_sample_image(data_2d)
        
        # 5. 电极布局可视化（可选）
        print("\n6. 电极布局可视化...")
        print("   显示电极位置图...")
        visualize_eeg_layout()
        
        print("\n🎉 演示完成！")
        print("\n使用说明:")
        print("- 使用 reshape_data_for_cnn() 函数转换你的数据")
        print("- 输入数据形状应为 (2, 5, 50, 62, 5)")
        print("- 输出数据形状为 (2, 5, 50, 5, 32, 32)")
        print("- 32x32 是默认的图像分辨率，可以通过 grid_res 参数调整")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
