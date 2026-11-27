# The single-GPU AlexNet implementation was used as inspiration before adapting it for the A9 part
# from the course material 
# 
import torch.nn as nn


class AlexNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, dropout=0.5):
        super().__init__()
        self.block1 = AlexNetBlock(in_channels, 64, pool=True)  # 64 x 16 x 16
        self.block2 = AlexNetBlock(64, 192, pool=True)  # 192 x 8 x 8
        self.block3 = AlexNetBlock(192, 384)  # 384 x 8 x 8
        self.block4 = AlexNetBlock(384, 256)  # 256 x 8 x 8
        self.block5 = AlexNetBlock(256, 256, pool=True)  # 256 x 4 x 4

        self.flatten = nn.Flatten()  # 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.classification_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.classification_layer(x)
