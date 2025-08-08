# src/eeg_mapper/model.py

import torch
import torch.nn as nn

class EEGToVideoPerceptionMapper(nn.Module):
    """
    将编码后的EEG特征映射到视频生成模型的条件嵌入空间（视频感知空间）。
    """
    def __init__(self, eeg_embedding_dim: int, video_perception_dim: int, hidden_dim: int = 4096, dropout: float = 0.5):
        """
        Args:
            eeg_embedding_dim (int): 输入的EEG特征嵌入维度。
                                     (例如，EEG分类器倒数第二层的输出维度)
            video_perception_dim (int): 目标视频感知空间的维度。
                                         (例如，WanT2V模型的条件输入维度)
            hidden_dim (int): MLP中间层的维度。
        """
        super(EEGToVideoPerceptionMapper, self).__init__()
        
        # 一个简单的多层感知机（MLP）作为映射器
        self.mapper = nn.Sequential(
            nn.Linear(eeg_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, video_perception_dim)
        )

    def forward(self, eeg_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_embeddings (torch.Tensor): 形状为 (batch_size, eeg_embedding_dim) 的EEG嵌入。

        Returns:
            torch.Tensor: 形状为 (batch_size, video_perception_dim) 的视频感知嵌入。
        """
        return self.mapper(eeg_embeddings)