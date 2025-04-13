import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakePPO(nn.Module):
    def __init__(self):
        super(SnakePPO, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        

        self.fc1 = nn.Linear(32 * 3 * 3, 128)  # 32 * 3 * 3 = 288，遠小於之前的參數量
        self.fc2 = nn.Linear(128, 64)
        
        self.policy_head = nn.Linear(64, 4)
        self.value_head = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        action_probs = F.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        
        return action_probs, state_value
