import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakePPO(nn.Module):
    def __init__(self, channel=4):
        super(SnakePPO, self).__init__()
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.policy_fc1 = nn.LazyLinear(256)
        self.policy_fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 4)

        self.value_fc1 = nn.LazyLinear(256)
        self.value_fc2 = nn.Linear(256, 256)
        self.value_fc3 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        policy_x = F.relu(self.policy_fc1(x))
        policy_x = F.relu(self.policy_fc2(policy_x))
        action_logits = self.policy_head(policy_x)

        value_x = F.relu(self.value_fc1(x))
        value_x = F.relu(self.value_fc2(value_x))
        value_x = F.relu(self.value_fc3(value_x))
        state_value = self.value_head(value_x)

        return action_logits, state_value


class SnakeDQN(nn.Module):
    def __init__(self, channel=4):
        super(SnakeDQN, self).__init__()
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.head(x)

        return q_values
