---
alwaysApply: true
---
my_EEG2Video/
├── 00_slice_eeg.ipynb                # [Notebook] 步骤1: 从原始.cnt档切片和预处理EEG数据
├── 01_extract_PSD_DE.ipynb           # [Notebook] 步骤2: 从切片后的EEG数据中提取PSD和DE特征
├── 02_train_eeg2label.ipynb          # [Notebook] 步骤3: 训练EEG到影片标签的分类模型
├── 03_video2text.ipynb               # [Notebook] 步骤4: 将视频转换为文字
├── 04_text_embedding.ipynb           # [Notebook] 步骤5: 将步骤4中的文字转换为embedding vector
├── 05_train_eeg_embedding.ipynb      # [Notebook] 步骤6: 训练EEG到语义嵌入空间的映射模型
|
├── data/                               # 存放所有原始和处理后的数据
│   ├── raw_eeg/                        # 存放原始的EEG数据 (.cnt格式)
│   ├── sliced_eeg/                     # 存放由`00_slice_eeg`处理后的数据
│   │   ├── watching/                   #   - 观看影片时的EEG数据片段 (.npy)
│   │   └── imaging/                    #   - 想像影片时的EEG数据片段 (.npy)
│   │
│   ├── PSD_DE/                         # 存放由`01_extract_PSD_DE`提取的特征
│   │   ├── watching/                   #   - 观看时的PSD/DE特征 (.npy)
│   │   └── imaging/                    #   - 想像时的PSD/DE特征 (.npy)
│   │
│   └── metadata/                       # 存放标签和文本嵌入等元数据
│       ├── GT_label.npy                # 影片的类别标签 (用于分类任务)
│       ├── subject_info.json           # 存放被试者信息
│       ├── text_embedding.pt           # 影片的文本语义嵌入 (用于映射任务)
│       └── video_descriptions_en_short.json # 存放影片的文本描述
|
├── models/                             # 存放模型定义的Python脚本
│   └── eeg2label.py                    # 包含glfnet模型的定义 (由02_...ipynb导入)
│
├── checkpoints/                        # 存放所有训练好的模型权重
│   ├── eeg2label/                      # 分类模型的权重
│   │   └── best_model.pth
│   └── eeg_embedding/                  # 语义嵌入模型的权重
│       └── best_model.pth
|
├── requirements.txt                    # 专案所需的Python依赖库
└── README.md                           # 专案总说明文件