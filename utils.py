import os
import glob
import numpy as np
import librosa
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 定义类别标签字典
LABEL_DICT = {
    'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4,
    'chips': 5, 'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9,
    'gummies': 10, 'ice-cream': 11, 'jelly': 12, 'noodles': 13, 'pickles': 14,
    'pizza': 15, 'ribs': 16, 'salmon': 17, 'soup': 18, 'wings': 19
}

LABEL_DICT_INV = {v: k for k, v in LABEL_DICT.items()}

def extract_features(parent_dir, sub_dirs=None, max_file=None, file_ext="*.wav"):
    """
    从音频文件中提取梅尔频谱特征
    
    Args:
        parent_dir: 音频文件根目录
        sub_dirs: 子目录列表，如果为None则处理parent_dir下的所有文件
        max_file: 每个类别最多处理的文件数，None表示处理所有文件
        file_ext: 文件扩展名
        
    Returns:
        features: 特征列表
        labels: 标签列表
    """
    features = []
    labels = []
    
    if sub_dirs is None:
        # 处理单个目录下的所有文件
        for fn in tqdm(glob.glob(os.path.join(parent_dir, file_ext))[:max_file]):
            try:
                # 使用默认参数加载音频
                X, sample_rate = librosa.load(fn, sr=None)
                # 提取梅尔频谱特征
                mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                features.append(mels)
                if os.path.basename(os.path.dirname(fn)) in LABEL_DICT:
                    labels.append(LABEL_DICT[os.path.basename(os.path.dirname(fn))])
            except Exception as e:
                print(f"Error processing file {fn}: {str(e)}")
                continue
    else:
        # 处理多个子目录下的文件
        for sub_dir in sub_dirs:
            for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
                try:
                    # 使用默认参数加载音频
                    X, sample_rate = librosa.load(fn, sr=None)
                    # 提取梅尔频谱特征
                    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                    features.append(mels)
                    labels.append(LABEL_DICT[sub_dir])
                except Exception as e:
                    print(f"Error processing file {fn}: {str(e)}")
                    continue
    
    return features, labels

def prepare_data(features, labels, test_size=0.25, random_state=1):
    """
    准备训练和测试数据
    
    Args:
        features: 特征列表
        labels: 标签列表
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        X_train, X_test, Y_train, Y_test: 训练集和测试集
    """
    # 转换为numpy数组
    X = np.vstack(features)
    Y = np.array(labels)
    
    # 重塑特征以适应CNN输入 (PyTorch格式: [batch, channels, height, width])
    X = X.reshape(-1, 1, 16, 8)
    
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    
    return X_train, X_test, Y_train, Y_test 