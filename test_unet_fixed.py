#!/usr/bin/env python3
"""
测试修复后的UNet模型加载功能
"""

import sys
import os
import traceback

# 添加models目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def test_unet_loading():
    """测试UNet模型加载功能"""
    print("🧪 Testing UNet model loading...")
    
    try:
        # 导入修复后的UNet模型
        print("📦 Importing UNet3DConditionModel...")
        from unet_fixed import UNet3DConditionModel
        
        print("✅ Successfully imported UNet3DConditionModel from unet_fixed.py")
        
        # 测试从预训练模型加载
        print("🔄 Testing model loading from CompVis/stable-diffusion-v1-4...")
        
        # 使用修复后的方法加载模型
        unet = UNet3DConditionModel.from_pretrained_2d(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="unet"
        )
        
        print("✅ Successfully loaded UNet model!")
        print(f"📊 Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unet_loading()
    if success:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)
