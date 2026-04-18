import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakePPO(nn.Module):
    def __init__(self):
        super(SnakePPO, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.LazyLinear(128)  
        self.fc2 = nn.Linear(128, 64)
        
        self.policy_head = nn.Linear(64, 4)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.policy_head(x)
        state_value = self.value_head(x)

        return action_logits, state_value
