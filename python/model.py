import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    # 黑子分布 - 白子分布 - 当前黑白 - 上一层黑子分布 - 上一层白子分布
    # 8 * 8 * 5

    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # (8 - 4) * (8 - 4) * 512

        self.link = nn.Sequential(
            nn.Linear(4 * 4 * 512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.output_p = nn.Linear(512, 64)
        self.output_v = nn.Linear(512, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.link(x)
        # v 的值域 范围 (-1, 1)
        p = F.log_softmax(self.output_p(x), dim=1)
        v = torch.tanh(self.output_v(x))
        return [p, v]
