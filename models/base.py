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
    def __init__(self, input_channels, board_shape=(8, 8), policy_size=4672, num_res_blocks=3, num_channels=32):
        super(BaseModel, self).__init__()
        self.input_channels = input_channels
        self.board_shape = board_shape
        self.policy_size = policy_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        board_area = board_shape[0] * board_shape[1]

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*[ResNetBlock(num_channels, num_channels) for _ in range(num_res_blocks)])

        self.plocy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_area, policy_size))

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_area, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        p = self.plocy_head(x)
        v = self.value_head(x)
        return p, v
