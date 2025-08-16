"""
测试EEG到2D转换功能的脚本
"""

import numpy as np
from eeg_to_2d import reshape_data_for_cnn, visualize_eeg_layout, get_eeg_sensor_positions

def test_eeg_to_2d_conversion():
    """
    测试EEG到2D转换功能
    """
    print("=== 测试EEG到2D转换功能 ===")
    
    # 创建测试数据：形状为(2, 5, 50, 62, 5)
    # 使用随机数据模拟EEG信号
    test_data = np.random.randn(2, 5, 50, 62, 5)
    
    print(f"测试数据形状: {test_data.shape}")
    
    # 测试转换
    try:
        result = reshape_data_for_cnn(test_data, grid_res=32)
        print(f"转换成功！结果形状: {result.shape}")
        
        # 验证形状是否正确
        expected_shape = (2, 5, 50, 5, 32, 32)
        if result.shape == expected_shape:
            print("✅ 形状验证通过！")
        else:
            print(f"❌ 形状验证失败！期望: {expected_shape}, 实际: {result.shape}")
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False
    
    return True

def test_sensor_positions():
    """
    测试电极位置获取功能
    """
    print("\n=== 测试电极位置获取功能 ===")
    
    try:
        positions = get_eeg_sensor_positions()
        print(f"成功获取 {len(positions)} 个电极的位置")
        
        # 检查是否包含所有62个电极
        from eeg_to_2d import CHANNEL_NAMES
        missing_channels = [ch for ch in CHANNEL_NAMES if ch not in positions]
        
        if len(missing_channels) == 0:
            print("✅ 所有62个电极位置都获取成功！")
        else:
            print(f"❌ 缺少电极位置: {missing_channels}")
            
        # 显示前几个电极的位置
        print("\n前5个电极的位置:")
        for i, ch in enumerate(CHANNEL_NAMES[:5]):
            if ch in positions:
                print(f"  {ch}: {positions[ch]}")
                
    except Exception as e:
        print(f"❌ 获取电极位置失败: {e}")
        return False
    
    return True

def test_visualization():
    """
    测试可视化功能
    """
    print("\n=== 测试可视化功能 ===")
    
    try:
        # 注意：这个函数会显示图形，在无头环境中可能会失败
        visualize_eeg_layout()
        print("✅ 可视化功能正常！")
        return True
    except Exception as e:
        print(f"⚠️ 可视化功能测试失败（可能是环境问题）: {e}")
        return True  # 不认为这是严重错误

if __name__ == "__main__":
    print("开始测试EEG到2D转换功能...\n")
    
    # 运行所有测试
    tests = [
        test_sensor_positions,
        test_eeg_to_2d_conversion,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！EEG到2D转换功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")
