"""
Simplified Gemini 2.5 Pro API Video Description Generator (English)

This script uses Google Gemini 2.5 Pro API to generate detailed English descriptions 
for video files. Simplified version without monitoring features.

Author: AI Assistant  
Dependencies: google-generativeai, opencv-python, pillow
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# if you are from China or some other countries where Google services are blocked, set the network proxy
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"

video2text_config = json.load(open("video2text_config.json", "r"))

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_video_processing_en.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeminiVideoDescriptorEN:
    """
    Simplified Gemini 2.5 Pro API Video Description Generator for English
    
    This class provides a streamlined interface for generating English video descriptions
    using Google's Gemini API.
    """
    
    def __init__(self, api_key: str, rate_limit_delay: float = 2.0):
        """
        Initialize the Gemini video description generator
        
        Args:
            api_key (str): Google Gemini API key
            rate_limit_delay (float): Delay between API calls to avoid rate limits
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        
        # 配置Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        logger.info("Gemini Video Description Generator (EN) initialized successfully")
    
    def generate_video_description(self, video_path: str) -> str:
        """
        Generate detailed English description for a single video
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Generated video description text in English
        """
        video_file = None
        try:
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return "Video file not found"
            
            # 获取视频文件大小（MB）
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"Uploading video {os.path.basename(video_path)} ({file_size:.1f}MB)...")
            
            # 上传视频文件到Gemini
            video_file = genai.upload_file(path=video_path)
            logger.info(f"Video uploaded successfully: {video_file.uri}")
            
            # 等待文件变为ACTIVE状态
            logger.info(f"Waiting for Google to process video file {os.path.basename(video_path)}...")
            max_wait_time = 300  # 最大等待5分钟
            wait_start = time.time()
            
            while True:
                try:
                    file_info = genai.get_file(name=video_file.name)
                    state = file_info.state.name
                    
                    if state == "ACTIVE":
                        logger.info(f"Video file {os.path.basename(video_path)} processing completed")
                        break
                    elif state == "FAILED":
                        logger.error(f"Video file {os.path.basename(video_path)} processing failed")
                        return f"Video file processing failed: {state}"
                    else:
                        elapsed = time.time() - wait_start
                        if elapsed > max_wait_time:
                            logger.error(f"Video file processing timeout ({max_wait_time}s)")
                            return f"Video file processing timeout: exceeded {max_wait_time}s"
                        
                        logger.debug(f"Video file status: {state}, continuing to wait... ({elapsed:.1f}s)")
                        time.sleep(3)  # 每3秒检查一次
                        
                except Exception as e:
                    logger.error(f"Error checking file status: {str(e)}")
                    time.sleep(5)
            
            # 构建英文提示词
            prompt = """
            Generate a concise, one-sentence description for this video, as if you were writing alt-text for an image. The description should be purely visual and capture the main scene and action in under 70 words.
            """
            
            # 调用Gemini API生成描述
            logger.info(f"Analyzing video {os.path.basename(video_path)}...")
            response = self.model.generate_content([prompt, video_file])
            
            # API调用延迟，避免超出速率限制
            time.sleep(self.rate_limit_delay)
            
            description = response.text.strip()
            logger.info(f"Description generation completed for video {os.path.basename(video_path)}")
            logger.debug(f"Generated description: {description[:100]}...")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating video description for {video_path}: {str(e)}")
            return f"Description generation failed: {str(e)}"
        
        finally:
            # 清理上传的文件（可选，Gemini会自动清理）
            if video_file:
                try:
                    pass  # genai.delete_file() 可能不存在
                except:
                    pass
    
    def get_video_files_by_groups(self, base_path: str) -> Dict[str, list]:
        """
        Get video files organized by groups
        
        Args:
            base_path (str): Base path containing group1-group5 folders
            
        Returns:
            Dict[str, list]: Dictionary of video file paths organized by group
        """
        video_files_by_group = {}
        base_path = Path(base_path)
        
        for group_num in range(1, 6):  # group1 到 group5
            group_name = f"group{group_num}"
            group_path = base_path / group_name
            
            if not group_path.exists():
                logger.warning(f"Group folder does not exist: {group_path}")
                video_files_by_group[group_name] = []
                continue
            
            # 获取该组中的所有.mp4文件
            video_files = list(group_path.glob("*.mp4"))
            video_files = [str(f) for f in video_files]
            
            logger.info(f"{group_name} found {len(video_files)} video files")
            video_files_by_group[group_name] = video_files
        
        return video_files_by_group
    
    def process_all_videos(self, base_path: str, output_dir: str = None) -> Dict[str, str]:
        """
        Process all video files and generate English descriptions
        
        Args:
            base_path (str): Base path containing group1-group5 folders
            output_dir (str): Output directory, defaults to current script directory
            
        Returns:
            Dict[str, str]: Dictionary mapping video filenames to descriptions
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取所有视频文件
        video_files_by_group = self.get_video_files_by_groups(base_path)
        
        # 构建完整的视频文件列表，按组顺序
        all_video_files = []
        for group_num in range(1, 6):
            group_name = f"group{group_num}"
            all_video_files.extend(video_files_by_group.get(group_name, []))
        
        logger.info(f"Total {len(all_video_files)} video files found")
        
        # 期望250个视频文件
        if len(all_video_files) != 250:
            logger.warning(f"Expected 250 video files, found {len(all_video_files)}")
        
        # 生成英文描述 - 使用字典格式，键为视频文件名
        descriptions_dict = {}
        failed_videos = []
        
        logger.info("Starting video description generation...")
        
        with tqdm(total=len(all_video_files), desc="Processing videos") as pbar:
            for i, video_path in enumerate(all_video_files):
                try:
                    video_name = os.path.basename(video_path)
                    logger.info(f"Processing video {i+1}/{len(all_video_files)}: {video_name}")
                    
                    description = self.generate_video_description(video_path)
                    descriptions_dict[video_name] = description
                    
                    # 定期保存中间结果
                    if (i + 1) % 10 == 0:
                        self._save_intermediate_results(descriptions_dict, output_dir, i + 1)
                    
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {str(e)}")
                    video_name = os.path.basename(video_path)
                    descriptions_dict[video_name] = f"Processing failed: {str(e)}"
                    failed_videos.append(video_path)
                
                pbar.update(1)
        
        # 保存最终结果
        self._save_final_results(descriptions_dict, failed_videos, output_dir)
        
        logger.info(f"Video description generation completed! Processed {len(descriptions_dict)} videos")
        if failed_videos:
            logger.warning(f"{len(failed_videos)} videos failed to process")
        
        return descriptions_dict
    
    def _save_intermediate_results(self, descriptions_dict: Dict[str, str], output_dir: str, count: int):
        """Save intermediate results"""
        intermediate_file = os.path.join(output_dir, f"video_descriptions_intermediate_{count}.json")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions_dict, f, ensure_ascii=False, indent=2)
        logger.debug(f"Intermediate results saved to: {intermediate_file}")
    
    def _save_final_results(self, descriptions_dict: Dict[str, str], failed_videos: list, output_dir: str):
        """Save final results to JSON format"""
        
        # 保存主要结果：{文件名: 描述} 格式的JSON文件
        json_file = os.path.join(output_dir, "video_descriptions_en.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"English video descriptions saved to: {json_file}")
        
        # 保存处理报告
        successful_count = len(descriptions_dict) - len(failed_videos)
        report_file = os.path.join(output_dir, "processing_report_en.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("English Video Description Generation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total videos: {len(descriptions_dict)}\n")
            f.write(f"Successfully processed: {successful_count}\n")
            f.write(f"Failed: {len(failed_videos)}\n")
            f.write(f"Success rate: {(successful_count / len(descriptions_dict) * 100):.2f}%\n\n")
            
            f.write("Output format:\n")
            f.write("- video_descriptions_en.json: {\"filename\": \"description\"} format\n")
            f.write("- Example: {\"1.mp4\": \"This video shows...\", \"2.mp4\": \"The video depicts...\"}\n\n")
            
            if failed_videos:
                f.write("Failed video files:\n")
                for video in failed_videos:
                    video_name = os.path.basename(video)
                    f.write(f"- {video_name}: {descriptions_dict.get(video_name, 'Unknown error')}\n")
        
        logger.info(f"Processing report saved to: {report_file}")


def main():
    """
    Main function: Execute video description generation task
    """
    # 配置参数
    API_KEY = os.getenv("google_api_key")  # 从环境变量获取API密钥
    VIDEO_BASE_PATH = r"D:\00 download\工作区\final\shuffled_renamed_videos"
    
    # 验证API密钥
    if not API_KEY:
        logger.error("Please set your Gemini API key first!")
        print("\nPlease follow these steps to set up API key:")
        print("1. Visit https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set the API key in your environment variables as 'google_api_key'")
        return
    
    # 验证视频路径
    if not os.path.exists(VIDEO_BASE_PATH):
        logger.error(f"Video base path does not exist: {VIDEO_BASE_PATH}")
        return
    
    try:
        # 创建英文视频描述生成器
        descriptor = GeminiVideoDescriptorEN(
            api_key=API_KEY,
            rate_limit_delay=2.0  # 2秒延迟，避免API限制
        )
        
        # 处理所有视频
        descriptions = descriptor.process_all_videos(VIDEO_BASE_PATH)
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Generated {len(descriptions)} English video descriptions")
        print(f"📁 Results saved to current directory")
        print(f"📝 Check processing_report_en.txt for detailed report")
        
    except Exception as e:
        logger.error(f"Error during program execution: {str(e)}")
        print(f"❌ Execution failed: {str(e)}")


if __name__ == "__main__":
    main()
