import torch
import torch.nn as nn

class SoundClassifier(nn.Module):
    def __init__(self, num_classes=20):
        """
        创建CNN模型用于声音分类
        
        Args:
            num_classes: 分类类别数，默认为20
        """
        super(SoundClassifier, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(0.1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

def create_model(num_classes=20):
    """
    创建并返回模型实例
    
    Args:
        num_classes: 分类类别数，默认为20
        
    Returns:
        模型实例
    """
    model = SoundClassifier(num_classes=num_classes)
    return model 