import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
    def __init__(self,state,actions):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(state,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,actions)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x