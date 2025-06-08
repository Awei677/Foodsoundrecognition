import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import create_model
from utils import extract_features, prepare_data, LABEL_DICT

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(train_dir, model_save_path='model.pth', epochs=20, batch_size=15, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练声音分类模型
    
    Args:
        train_dir: 训练数据目录
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        device: 训练设备（GPU/CPU）
    """
    # 获取所有类别目录
    sub_dirs = list(LABEL_DICT.keys())
    
    # 提取特征
    print("Extracting features...")
    features, labels = extract_features(train_dir, sub_dirs=sub_dirs)
    
    # 准备数据
    print("Preparing data...")
    X_train, X_test, Y_train, Y_test = prepare_data(features, labels)
    
    # 创建数据加载器
    train_dataset = AudioDataset(X_train, Y_train)
    test_dataset = AudioDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # 训练模型
    print("Training model...")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')
        
        # 保存最佳模型
        if val_correct/val_total > best_acc:
            best_acc = val_correct/val_total
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
    
    return model

if __name__ == "__main__":
    # 设置训练参数
    TRAIN_DIR = "./train_sample"  # 训练数据目录
    MODEL_SAVE_PATH = "sound_classifier.pth"  # 模型保存路径
    
    # 训练模型
    model = train(
        train_dir=TRAIN_DIR,
        model_save_path=MODEL_SAVE_PATH,
        epochs=20,
        batch_size=15
    ) 