import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes=10, ndf=64):
        super(FCDiscriminator, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.trans_1 = conv_trans_block_3d(1, 1, act_fn)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.classifier(x)
        # x=x.reshape(-1,4*4*2)
        # x=self.liner(x)
        x = self.sigmoid(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


