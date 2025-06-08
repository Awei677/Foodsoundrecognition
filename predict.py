import os
import torch
import numpy as np
import pandas as pd
from model import create_model
from utils import extract_features, LABEL_DICT

def predict_audio(model_path, audio_dir, output_file='predictions.csv', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用训练好的模型预测音频文件类别
    
    Args:
        model_path: 模型文件路径
        audio_dir: 待预测音频文件目录
        output_file: 预测结果保存路径
        device: 预测设备（GPU/CPU）
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 检查音频目录是否存在
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # 检查音频文件
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        raise ValueError(f"No .wav files found in {audio_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # 加载模型
    print("Loading model...")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 提取特征
    print("Extracting features...")
    features, _ = extract_features(audio_dir)
    
    if not features:
        raise ValueError("No features extracted from audio files")
    
    # 准备数据
    X = np.array(features)
    # 重塑特征以适应CNN输入 (PyTorch格式: [batch, channels, height, width])
    X = X.reshape(-1, 1, 16, 8)
    X = torch.FloatTensor(X).to(device)
    
    # 预测
    print("Making predictions...")
    predictions = []
    with torch.no_grad():
        outputs = model(X)
        _, predicted = outputs.max(1)
        predictions = predicted.cpu().numpy()
    
    # 创建预测结果DataFrame
    results = pd.DataFrame({
        'audio_file': audio_files,
        'predicted_class': [list(LABEL_DICT.keys())[pred] for pred in predictions]
    })
    
    # 保存预测结果
    print(f"Saving predictions to {output_file}...")
    results.to_csv(output_file, index=False)
    
    return results

if __name__ == "__main__":
    # 设置预测参数
    MODEL_PATH = "sound_classifier.pth"  # 模型文件路径
    AUDIO_DIR = "test_a"  # 待预测音频文件目录
    OUTPUT_FILE = "predictions.csv"  # 预测结果保存路径
    
    try:
        # 进行预测
        results = predict_audio(
            model_path=MODEL_PATH,
            audio_dir=AUDIO_DIR,
            output_file=OUTPUT_FILE
        )
        
        # 打印预测结果
        print("\nPrediction Results:")
        print(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure:")
        print("1. The model file exists at:", MODEL_PATH)
        print("2. The audio directory exists at:", AUDIO_DIR)
        print("3. There are .wav files in the audio directory") 