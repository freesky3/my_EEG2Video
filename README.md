You can now train our project on colab!🥰

[02_train_eeg2label.ipynb - Colab](https://colab.research.google.com/drive/1URpVY6gb1OmBvGQtd1_IxHZ_8HD5Ph_M)

[05_train_eeg_embedding.ipynb - Colab](https://colab.research.google.com/drive/1ApYzv9JYhV7Wn3Fw8y6aFX2LFyzUpmCP)

[07_Seq2seq.ipynb - Colab](https://colab.research.google.com/drive/1KjVduZO0yjlw34m8Y3IHY3QPMepqpM4y#scrollTo=65MJO-6HPPco) [not usable yet]

[09_add_noise.ipynb - Colab](https://colab.research.google.com/drive/10FWIWK1aLPPeVVEFg_5nmpAOMuYIUO7R#scrollTo=944bc274)

[10_train_finetune_videodiffusion.ipynb - Colab](https://colab.research.google.com/drive/1qK6bq9PbcRuYZXElH93SI5_zhwK8TZQB#scrollTo=jpmYTcxzTS24) [not usable yet]

[11_inference_eeg2video.ipynb - Colab](https://colab.research.google.com/drive/10AA62jsDE5o7-3JMEehfnkMdF24n_Lid) [not usable yet]

[wan_embedding2video_Ampere.ipynb - Colab](https://colab.research.google.com/drive/1UVgchehpL8WLZlQ5leAuDD23aHN7esvC#scrollTo=YYzcq3ka2Q_1)

[wan_embedding2video_not_Ampere.ipynb - Colab](https://colab.research.google.com/drive/1nRFA-fResNHrAUDlTl8xKeYZN8xY7vOl#scrollTo=6kGYIxG_2X_e)

use python=3.11.13

you need go to [pytorch](https://pytorch.org/#:~:text=and%20easy%20scaling.-,Install%20PyTorch,-Select%20your%20preferences) to install torch and torchvision first. 

```txt
my_EEG2Video/
├── 00_slice_eeg.ipynb                # [Notebook] 步骤 0: 从原始.cnt文件切片和预处理EEG数据
├── 01_extract_PSD_DE.ipynb           # [Notebook] 步骤 1: 从切片后的EEG数据中提取PSD和DE特征
├── 02_train_eeg2label.ipynb          # [Notebook] 步骤 2: (可选) 训练EEG到视频标签的分类模型
├── 03_video2text.ipynb               # [Notebook] 步骤 3: 使用多模态大模型将视频转换为文字描述
├── 04_text_embedding.ipynb           # [Notebook] 步骤 4: 使用CLIP模型将文字描述转换为语义嵌入 (Text Embedding)
├── 05_train_eeg_embedding.ipynb      # [Notebook] 步骤 5: 训练EEG到语义嵌入空间的映射模型
├── 06_video2latent.ipynb             # [Notebook] 步骤 6: 使用VAE将视频帧编码为潜空间向量 (Latent)
├── 07_Seq2seq.ipynb                  # [Notebook] 步骤 7: 训练Seq2Seq模型，将EEG序列预测为视频潜空间向量序列
├── 08_opt_flow.ipynb                 # [Notebook] 步骤 8: 计算视频的光流分数，量化动态程度
├── 09_add_noise.ipynb                # [Notebook] 步骤 9: (DANA模块) 根据光流为潜空间向量添加动态噪声
├── 10_train_finetune_videodiffusion.ipynb # [Notebook] 步骤 10: 微调视频扩散模型 (Tune-A-Video)
├── 11_inference_eeg2video.ipynb      # [Notebook] 步骤 11: 最终推理流程，从EEG生成视频
├── 12_run_metrics.ipynb              # [Notebook] 步骤 12: 计算生成视频的评估指标 (SSIM, FID等)
|
├── data/                               # [目录] 存放所有原始和处理后的数据 (被.gitignore忽略)
│   ├── raw_eeg/                        #  - 存放原始的EEG数据 (.cnt格式)
│   ├── sliced_eeg/                     #  - [产出自 00] 存放切片后的EEG数据
│   │   ├── watching/                   #    - 观看影片时的EEG数据片段 (.npy)
│   │   └── imaging/                    #    - 想象影片时的EEG数据片段 (.npy)
│   ├── PSD_DE/                         #  - [产出自 01] 存放提取的PSD/DE特征
│   │   ├── watching/
│   │   └── imaging/
│   ├── videos/                         #  - 存放原始的影片文件 (.mp4)
│   └── metadata/                       #  - 存放标签和各种中间产出
│       ├── GT_label.npy                #    - 视频的类别标签 (用于分类任务)
│       ├── video_descriptions_en_short.json     #    - [产出自 03] 视频的文本描述
│       ├── text_embedding.pt           #    - [产出自 04] 文本的语义嵌入
│       ├── videos_latents.pt           #    - [产出自 06] 视频的潜空间向量 (Seq2Seq目标)
│       ├── seq2seq_predicted_latents.pt #   - [产出自 07] Seq2Seq模型预测的潜空间向量
│       ├── prompt_ids.pt                #    - [产出自 08] 提示词ID
│       ├── subject_info.json            #    - [产出自 08] 受试者信息
│       ├── videos_frames.npy             #    - [产出自 06] 视频的帧
│       ├── optical_flow_scores.npy       #    - [产出自 08] 视频的光流分数
│       └── noise_videos_latents.pt       #    - [产出自 09] 添加了DANA噪声的潜空间向量
|
├── checkpoints/                        # [目录] 存放所有训练好的模型权重 (被.gitignore忽略)
│   ├── eeg2label/                      #  - [产出自 02] 分类模型权重
│   ├── eeg_embedding/                  #  - [产出自 05] 语义嵌入模型权重
│   ├── seq2seq/                        #  - [产出自 07] 序列到序列模型权重
│   └── video_diffusion_finetuned/      #  - [产出自 10] 微调后的视频扩散模型
|
├── models/                             # [目录] 存放模型架构的Python脚本
│   ├── attention.py                    #  - 3D 注意力模块 (Tune-A-Video核心)
│   ├── resnet.py                       #  - 3D ResNet 模块
│   ├── unet_blocks.py                  #  - 3D UNet 的构建块
│   ├── unet.py                         #  - 完整的 3D UNet 模型定义
│   └── pipeline_tuneavideo.py          #  - 自定义的视频生成 Pipeline
|
├── utils/                              # [目录] 存放可重用的工具函数和类
│   ├── dataset.py                      #  - PyTorch Dataset 类的定义
│   └── util.py                         #  - 辅助函数 (如保存视频网格)
|
├── .gitignore                          # Git忽略规则 (忽略 data/ 和 checkpoints/ 目录)
├── requirements.txt                    # 项目所需的Python依赖库
└── README.md                           # 项目总说明文件
```

