"""
此模块为视频扩散模型定义了 PyTorch 数据集类。

它提供了两个主要的类：
- TuneMultiVideoDataset: 用于加载和处理多个视频及其对应的文本提示。
- TuneAVideoDataset: TuneMultiVideoDataset 的一个特例，专门用于对单个视频进行微调。
"""

from typing import Dict, List, Union

import decord
import torch
from einops import rearrange
from torch.utils.data import Dataset

# 将 decord 的默认后端设置为 PyTorch，以便直接输出 torch.Tensor
decord.bridge.set_bridge('torch')


class TuneMultiVideoDataset(Dataset):
    """一个用于加载多个视频及其对应提示的数据集。

    此数据集从磁盘读取视频文件，对帧进行采样和预处理，
    并将其与预先分词好的文本提示配对，为视频扩散模型的训练做准备。

    Attributes:
        video_paths (List[str]): 视频文件的路径列表。
        prompts (List[str]): 每个视频对应的文本提示列表。
        prompt_ids (torch.Tensor): 预先分词并编码的文本提示ID。
        width (int): 视频帧的目标宽度。
        height (int): 视频帧的目标高度。
        n_sample_frames (int): 从每个视频中采样的帧数。
        sample_start_idx (int): 采样的起始帧索引。
        sample_frame_rate (int): 采样时帧之间的间隔。
    """

    def __init__(
        self,
        video_path: List[str],
        prompt: List[str],
        width: int = 1920,
        height: int = 1080,
        n_sample_frames: int = 15,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 4,
    ):
        """初始化 TuneMultiVideoDataset。

        Args:
            video_path (List[str]): 视频文件的路径列表。
            prompt (List[str]): 与每个视频对应的文本提示列表。
            width (int, optional): 视频帧的宽度。默认为 1920。
            height (int, optional): 视频帧的高度。默认为 1080。
            n_sample_frames (int, optional): 要采样的帧数。默认为 15。
            sample_start_idx (int, optional): 采样的起始帧索引。默认为 0。
            sample_frame_rate (int, optional): 采样帧率。默认为 4。
        """
        self.video_paths = video_path
        self.prompts = prompt
        self.prompt_ids = None  # 将在训练脚本中预先计算和赋值

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self) -> int:
        """返回数据集中视频的总数。"""
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        根据索引获取一个数据样本。

        Args:
            index (int): 数据样本的索引。

        Returns:
            Dict[str, torch.Tensor]: 包含 "pixel_values" 和 "prompt_ids" 的字典。
        """
        # 加载视频并调整大小
        video_reader = decord.VideoReader(
            self.video_paths[index], width=self.width, height=self.height
        )
        video_length = len(video_reader)

        # 生成采样帧的索引
        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)
        start_idx = self.sample_start_idx
        end_idx = start_idx + clip_length
        sample_index = torch.linspace(start_idx, end_idx - 1, self.n_sample_frames).long()

        # 获取视频帧
        video = video_reader.get_batch(sample_index)

        # 调整维度顺序 (f, h, w, c) -> (f, c, h, w)
        video = rearrange(video, "f h w c -> f c h w")

        # 将像素值归一化到 [-1, 1]
        pixel_values = (video / 127.5) - 1.0

        example = {
            "pixel_values": pixel_values,
            "prompt_ids": self.prompt_ids[index]
        }
        return example


class TuneAVideoDataset(TuneMultiVideoDataset):
    """一个专门用于对单个视频进行微调的数据集。

    这是 TuneMultiVideoDataset 的一个特例，它接收单个视频路径和提示，
    并将其封装在列表中，以便重用多视频数据集的处理逻辑。
    `__len__` 方法被重写以始终返回 1。
    """

    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        n_sample_frames: int = 15,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 4,
    ):
        """初始化 TuneAVideoDataset。

        Args:
            video_path (str): 单个视频文件的路径。
            prompt (str): 对应的单个文本提示。
            width (int, optional): 视频帧的宽度。默认为 1920。
            height (int, optional): 视频帧的高度。默认为 1080。
            n_sample_frames (int, optional): 要采样的帧数。默认为 15。
            sample_start_idx (int, optional): 采样的起始帧索引。默认为 0。
            sample_frame_rate (int, optional): 采样帧率。默认为 4。
        """
        # 将单个路径和提示转换为列表，以调用父类的构造函数
        super().__init__(
            video_path=[video_path],
            prompt=[prompt],
            width=width,
            height=height,
            n_sample_frames=n_sample_frames,
            sample_start_idx=sample_start_idx,
            sample_frame_rate=sample_frame_rate,
        )

    def __len__(self) -> int:
        """重写长度方法，对于单个视频微调，数据集长度始终为 1。"""
        return 1