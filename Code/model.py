import torch
import torch.nn as nn


class Temporal_Module(nn.Module):
    def __init__(self):
        super(Temporal_Module, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=3, batch_first=True)
        self.upconv1 = nn.ConvTranspose2d(50, 256, kernel_size=4, stride=4)
        self.conv1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 512, kernel_size=4, stride=4)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.upconv1(x)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.upconv2(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))

        return x


class Visual_Module(nn.Module):
    def __init__(self):
        super(Visual_Module, self).__init__()
        self.temporal_module = Temporal_Module()

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv1 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, s):
        x = x.unsqueeze(1)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(self.pool(x1))
        x3 = self.encoder_3(self.pool(x2))
        x4 = self.encoder_4(self.pool(x3))
        x5 = self.encoder_5(self.pool(x4))

        s = self.temporal_module(s)
        x_s = torch.cat([x5, s], dim=1)

        x6 = self.upconv1(x_s)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.decoder_1(x6)

        x7 = self.upconv2(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.decoder_2(x7)

        x8 = self.upconv3(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.decoder_3(x8)

        x9 = self.upconv4(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.decoder_4(x9)

        out = self.output_layer(x9)
        out = out.squeeze(1)

        return out
