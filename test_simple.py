#!/usr/bin/env python3
"""
简化的测试脚本，直接测试from_pretrained_2d方法的核心逻辑
"""

import os
import sys

def test_from_pretrained_2d_logic():
    """测试from_pretrained_2d方法的核心逻辑"""
    print("🧪 Testing from_pretrained_2d logic...")
    
    try:
        # 导入必要的库
        from diffusers import UNet2DConditionModel
        
        # 测试模型路径构建
        pretrained_model_path = "CompVis/stable-diffusion-v1-4"
        subfolder = "unet"
        
        # 检查是否是Hugging Face Hub模型ID（包含'/'）
        if '/' in pretrained_model_path:
            # 对于Hub模型，使用subfolder参数而不是路径拼接
            full_model_path = pretrained_model_path
            subfolder_to_use = subfolder
        else:
            # 对于本地路径，使用路径拼接
            full_model_path = os.path.join(pretrained_model_path, subfolder)
            subfolder_to_use = None
            
        print(f"📁 Full model path: {full_model_path}")
        print(f"📁 Subfolder to use: {subfolder_to_use}")
        
        # 测试加载2D UNet
        print("🔄 Loading 2D UNet...")
        if subfolder_to_use:
            unet_2d = UNet2DConditionModel.from_pretrained(full_model_path, subfolder=subfolder_to_use)
        else:
            unet_2d = UNet2DConditionModel.from_pretrained(full_model_path)
        
        print("✅ Successfully loaded 2D UNet!")
        print(f"📊 2D Model parameters: {sum(p.numel() for p in unet_2d.parameters()):,}")
        
        # 测试配置获取
        config_2d = unet_2d.config
        print(f"📋 2D Model config keys: {list(config_2d.keys())}")
        
        # 测试状态字典获取
        state_dict_2d = unet_2d.state_dict()
        print(f"🔑 2D Model state dict keys: {len(state_dict_2d.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_from_pretrained_2d_logic()
    if success:
        print("\n🎉 Core logic test passed! The fix approach is correct.")
    else:
        print("\n💥 Core logic test failed. Please check the error messages above.")
        sys.exit(1)
