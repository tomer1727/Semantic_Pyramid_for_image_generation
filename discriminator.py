import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> Relu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride),
                                   nn.BatchNorm2d(num_filters),
                                   nn.LeakyReLU(0.2, inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv_block1 = ConvBlock(in_channels, in_channels, 1, padding=0)
        self.conv_block2 = ConvBlock(in_channels, in_channels, 3, stride=2)
        self.conv_block3 = ConvBlock(in_channels, out_channels, 1, padding=0, activation='none')
        self.skip_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(out_channels))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.skip_block(residual)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x += residual
        x = self.activation(x)
        return x


class SelfAttentionLayer(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttentionLayer, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.res_block1 = ResidualBlock(64, 256)
        self.attn1 = SelfAttentionLayer(in_dim=256)
        self.res_block2 = ResidualBlock(256, 512)
        self.res_block3 = ResidualBlock(512, 1024)
        self.res_block4 = ResidualBlock(1024, 2048)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=2048, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_block1(x)
        x, _ = self.attn1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
