import torch.nn as nn

#res net block for the base model
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity 
        out = self.relu(out)
        return out

class BaseModel(nn.Module):
    def __init__(self, input_channels, num_res_blocks=19):
        super(BaseModel, self).__init__()
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks

        self.stem = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.backbone = nn.Sequential(*[ResNetBlock(256, 256) for _ in range(num_res_blocks)])

        self.plocy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 8 * 8 * 73)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),  
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.plocy_head(x)
        return x
