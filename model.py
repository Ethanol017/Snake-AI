import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakePPO(nn.Module):
    def __init__(self):
        super(SnakePPO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 18 * 18, 256)  
        self.fc2 = nn.Linear(256, 128)

        # 策略輸出（行動機率）
        self.policy_head = nn.Linear(128, 4)  

        # 值函數輸出（估計當前狀態的價值）
        self.value_head = nn.Linear(128, 1)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_probs = F.softmax(self.policy_head(x), dim=-1)  # 行動機率
        state_value = self.value_head(x)  # 狀態值
        
        return action_probs, state_value
