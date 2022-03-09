import torch
import torch.nn as nn


class bottleneck(nn.Module):
    def __init__(self, in_channels, first_channels):
        super(bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=first_channels*2, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(first_channels*2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=first_channels*2, out_channels=first_channels*4, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(first_channels*4)

        self.conv3 = nn.Conv1d(in_channels=first_channels*4, out_channels=first_channels*8, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm1d(first_channels*8)

        self.down_sample = nn.Conv1d(in_channels=in_channels, out_channels=first_channels*8, kernel_size=1, padding=0, stride=1)
        self.bn_down = nn.BatchNorm1d(first_channels*8)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        out1 = self.down_sample(identity)
        out1 = self.bn_down(out1)

        out1 = x + out1
        out1 = self.relu(out1)
        return out1


class MeACP(nn.Module):
    def __init__(self):
        super(MeACP, self).__init__()
        # 96个参数
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

        # 234,144个参数
        self.layer1 = bottleneck(16, 16)
        self.layer2 = bottleneck(128, 32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)

        # 553,024个参数
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc2 = nn.Linear(128, 64)

        # 129个参数
        self.output_fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # conv

        out1 = self.layer1(x)
        out1 = self.layer2(out1)
        out1 = self.avgpool(out1).squeeze(2)
        out1 = self.fc1(out1)

        # lstm
        b, c, d = x.shape
        out2 = x.view(b, d, c)
        output, (h_n, c_n) = self.lstm(out2)
        out2 = self.fc2(h_n[-1])

        feature = torch.cat((out1, out2), 1)
        out = self.output_fc(feature)
        out = self.sigmoid(out.squeeze(1))
        return out


if __name__ == '__main__':
    model = MeACP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')