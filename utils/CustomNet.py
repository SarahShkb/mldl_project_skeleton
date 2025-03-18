import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 200)  # 200 is the number of classes
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer



        return x