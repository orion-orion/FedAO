import torch
import torch.nn.functional as F


class ConvNet(torch.nn.Module):
    def __init__(self, input_size, channels, num_classes):
        super(ConvNet, self).__init__()
        # 但
        self.conv1 = torch.nn.Conv2d(channels, 32, 5) #输入通道，输出通道，卷积核大小
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        
        if input_size == 28:
            self.fc1 = torch.nn.Linear(1024, 2048) 
            self.input_size = 28
            self.output = torch.nn.Linear(2048, num_classes) #62

        elif input_size == 32:
            self.fc1 = torch.nn.Linear(64 * 5 * 5, 2048) # 10
            self.input_size = 32
            self.output = torch.nn.Linear(2048, num_classes) #10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.input_size == 28:
            x = x.view(-1, 1024)
        elif self.input_size == 32:
            x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x