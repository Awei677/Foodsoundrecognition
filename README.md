# 食物声音识别系统

这是一个基于深度学习的食物声音识别系统，可以识别20种不同食物的声音。系统使用梅尔频谱图（Mel Spectrogram）作为特征，采用卷积神经网络（CNN）进行分类。

## 功能特点

- 支持20种食物声音的识别
- 使用梅尔频谱图进行特征提取
- 基于 PyTorch 实现的 CNN 模型
- 支持模型训练和预测
- 提供详细的训练过程监控
- 支持批量预测并输出 CSV 格式结果

## 环境要求

- Python 3.8 或更高版本
- PyTorch 2.0.0 或更高版本
- librosa 0.8.0 或更高版本
- numpy
- pandas
- scikit-learn
- tqdm

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd Foodsoundrecognition
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
Foodsoundrecognition/
├── model.py          # CNN 模型定义
├── utils.py          # 工具函数（特征提取、数据处理）
├── train.py          # 模型训练脚本
├── predict.py        # 模型预测脚本
├── requirements.txt  # 项目依赖
└── README.md         # 项目说明文档
```

## 使用方法

### 训练模型

1. 准备训练数据：
   - 将训练音频文件放在 `train_sample` 目录下
   - 每个类别的音频文件放在对应的子目录中

2. 运行训练脚本：
```bash
python train.py
```

训练过程中会显示：
- 训练损失和准确率
- 验证损失和准确率
- 最佳模型保存信息

### 预测

1. 准备测试数据：
   - 将待预测的音频文件放在 `test_a` 目录下

2. 运行预测脚本：
```bash
python predict.py
```

预测结果将保存在 `predictions.csv` 文件中，包含：
- 音频文件路径
- 预测的食物类别

## 支持的类别

系统可以识别以下20种食物的声音：
1. aloe（芦荟）
2. burger（汉堡）
3. cabbage（卷心菜）
4. candied_fruits（蜜饯）
5. carrots（胡萝卜）
6. chips（薯片）
7. chocolate（巧克力）
8. drinks（饮料）
9. fries（薯条）
10. grapes（葡萄）
11. gummies（软糖）
12. ice-cream（冰淇淋）
13. jelly（果冻）
14. noodles（面条）
15. pickles（泡菜）
16. pizza（披萨）
17. ribs（排骨）
18. salmon（三文鱼）
19. soup（汤）
20. wings（鸡翅）

## 模型架构

系统使用了一个简单的 CNN 模型，包含：
- 2个卷积层（带 ReLU 激活和最大池化）
- Dropout 层（防止过拟合）
- 2个全连接层
- 输出层（20个类别）

## 注意事项

1. 确保音频文件格式为 WAV
2. 训练数据需要按类别组织在子目录中
3. 预测时确保模型文件 `sound_classifier.pth` 存在
4. 建议使用 GPU 进行训练（如果可用）

## 性能指标

在测试集上的表现：
- 训练准确率：约 87%
- 验证准确率：约 43%

## 未来改进

1. 增加数据增强方法
2. 优化模型架构
3. 添加更多评估指标
4. 支持更多音频格式
5. 添加模型解释功能
