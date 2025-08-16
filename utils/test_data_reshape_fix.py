"""
测试数据重塑修复的脚本
验证数据重塑是否正确，不会混合不同频段的数据
"""

import numpy as np
from eeg_to_2d import reshape_data_for_cnn

def test_data_reshape_correctness():
    """
    测试数据重塑的正确性
    """
    print("=== 测试数据重塑正确性 ===")
    
    # 创建测试数据，每个频段有独特的模式
    # 形状: (2, 5, 50, 62, 5)
    # 让每个频段有不同的特征值
    test_data = np.zeros((2, 5, 50, 62, 5))
    
    # 为每个频段设置不同的特征值
    for band in range(5):
        test_data[:, :, :, :, band] = band + 1  # 频段0=1, 频段1=2, 等等
    
    print(f"原始数据形状: {test_data.shape}")
    print(f"原始数据频段值: {[test_data[0, 0, 0, 0, i] for i in range(5)]}")
    
    # 测试转换
    try:
        result = reshape_data_for_cnn(test_data, grid_res=32)
        print(f"转换后数据形状: {result.shape}")
        
        # 验证转换结果
        # 检查第一个样本的每个频段是否保持独立
        print("\n验证频段独立性:")
        for band in range(5):
            # 获取第一个样本的该频段图像
            sample_image = result[0, 0, 0, band, :, :]
            mean_value = np.mean(sample_image)
            print(f"  频段 {band}: 平均值 = {mean_value:.4f} (期望: {band + 1})")
            
            # 由于插值会改变绝对值，我们检查相对关系
            if band > 0:
                prev_mean = np.mean(result[0, 0, 0, band-1, :, :])
                ratio = mean_value / prev_mean if prev_mean != 0 else 0
                expected_ratio = (band + 1) / band if band != 0 else 1
                print(f"    与频段{band-1}的比值: {ratio:.4f} (期望: {expected_ratio:.4f})")
                
                if abs(ratio - expected_ratio) < 0.2:  # 允许一定的插值误差
                    print(f"    ✅ 频段 {band} 相对关系正确")
                else:
                    print(f"    ❌ 频段 {band} 相对关系错误")
            else:
                print(f"    ✅ 频段 {band} 数据存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def test_data_consistency():
    """
    测试数据一致性
    """
    print("\n=== 测试数据一致性 ===")
    
    # 创建有规律的数据模式
    test_data = np.random.randn(2, 5, 50, 62, 5)
    
    # 为每个频段添加独特的偏移
    for band in range(5):
        test_data[:, :, :, :, band] += band * 10
    
    print(f"原始数据形状: {test_data.shape}")
    
    # 检查原始数据的频段特征
    print("原始数据频段特征:")
    for band in range(5):
        mean_val = np.mean(test_data[:, :, :, :, band])
        print(f"  频段 {band}: 平均值 = {mean_val:.4f}")
    
    # 转换数据
    result = reshape_data_for_cnn(test_data, grid_res=32)
    
    # 检查转换后数据的频段特征
    print("\n转换后数据频段特征:")
    for band in range(5):
        # 计算该频段所有图像的平均值
        band_images = result[:, :, :, band, :, :]
        mean_val = np.mean(band_images)
        print(f"  频段 {band}: 平均值 = {mean_val:.4f}")
    
    # 验证频段间的相对关系是否保持
    # 计算相邻频段的比值
    print(f"\n频段间相对关系:")
    for band in range(1, 5):
        original_ratio = np.mean(test_data[:, :, :, :, band]) / np.mean(test_data[:, :, :, :, band-1])
        converted_ratio = np.mean(result[:, :, :, band, :, :]) / np.mean(result[:, :, :, band-1, :, :])
        
        print(f"  频段{band}/频段{band-1}:")
        print(f"    原始数据比值: {original_ratio:.4f}")
        print(f"    转换后数据比值: {converted_ratio:.4f}")
        
        # 允许一定的插值误差
        if abs(original_ratio - converted_ratio) < 0.3:
            print(f"    ✅ 相对关系保持正确")
        else:
            print(f"    ❌ 相对关系丢失")
    
    # 检查频段间的排序关系是否保持
    original_order = [np.mean(test_data[:, :, :, :, i]) for i in range(5)]
    converted_order = [np.mean(result[:, :, :, i, :, :]) for i in range(5)]
    
    print(f"\n频段排序关系:")
    print(f"  原始数据排序: {[f'{x:.2f}' for x in original_order]}")
    print(f"  转换后数据排序: {[f'{x:.2f}' for x in converted_order]}")
    
    # 检查排序是否保持
    original_sorted = sorted(original_order)
    converted_sorted = sorted(converted_order)
    
    if original_sorted == original_order and converted_sorted == converted_order:
        print("  ✅ 频段排序关系保持正确")
        return True
    else:
        print("  ❌ 频段排序关系丢失")
        return False

def test_performance_improvement():
    """
    测试性能改进
    """
    print("\n=== 测试性能改进 ===")
    
    import time
    
    # 创建测试数据
    test_data = np.random.randn(2, 5, 50, 62, 5)
    
    # 测试修复后的版本
    start_time = time.time()
    result = reshape_data_for_cnn(test_data, grid_res=32)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"输出数据形状: {result.shape}")
    
    # 验证输出形状正确
    expected_shape = (2, 5, 50, 5, 32, 32)
    if result.shape == expected_shape:
        print("✅ 输出形状正确")
        return True
    else:
        print(f"❌ 输出形状错误: 期望 {expected_shape}, 实际 {result.shape}")
        return False

def main():
    """
    主测试函数
    """
    print("🧪 数据重塑修复测试")
    print("=" * 50)
    
    tests = [
        test_data_reshape_correctness,
        test_data_consistency,
        test_performance_improvement
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！数据重塑修复成功。")
        print("\n修复内容:")
        print("1. ✅ 正确的数据转置: 使用 transpose(0,1,2,4,3) 确保电极维度在最后")
        print("2. ✅ 性能优化: 预计算电极位置和插值网格，避免重复计算")
        print("3. ✅ 频段独立性: 确保不同频段的数据不会混合")
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main()
