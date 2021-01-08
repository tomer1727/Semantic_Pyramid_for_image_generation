import torch
import torch.nn as nn


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


class ConvBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> Relu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride)),
                                   nn.BatchNorm2d(num_filters),
                                   nn.LeakyReLU(0.2, inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs

    def init_weights(self):
        self.block.apply(init_weights)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv_block1 = ConvBlock(in_channels, in_channels, 1, padding=0)
        self.conv_block2 = ConvBlock(in_channels, out_channels, 3, stride=2, padding=1, activation='none')
        # self.conv_block3 = ConvBlock(in_channels, out_channels, 1, padding=0, activation='none')
        self.skip_block = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False)),
                                        nn.BatchNorm2d(out_channels))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # residual = x.clone()
        # if self.in_channels != self.out_channels:
        residual = self.skip_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # x = self.conv_block3(x)
        x += residual
        x = self.activation(x)
        return x
    
    def init_weights(self):
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.skip_block.apply(init_weights)


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
        self.first_conv = nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2))
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv_block1 = ConvBlock(64, 64, 4, stride=2, padding=1)
        # self.attn1 = SelfAttentionLayer(in_dim=64)
        self.conv_block2 = ConvBlock(64, 128, 4, stride=2, padding=1)
        self.conv_block3 = ConvBlock(128, 256, 4, stride=2, padding=1)
        self.conv_block4 = ConvBlock(256, 512, 4, stride=2, padding=1)
        self.conv_block5 = ConvBlock(512, 256, 3, stride=2, padding=1)
        # self.conv1 = nn.utils.spectral_norm(nn.Conv2d(2048, 256, 3, stride=2, padding=1))
        self.last_conv = nn.Conv2d(256, 1, 4)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.flat = nn.Flatten()
        # self.fc = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.relu(x)
        x = self.conv_block1(x)
        # x, _ = self.attn1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        # x = self.conv1(x)
        x = self.last_conv(x)
        # x = self.avg_pool(x)
        # x = self.flat(x)
        # x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def init_weights(self):
        self.first_conv.apply(init_weights)
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.conv_block3.init_weights()
        self.conv_block4.init_weights()
        self.conv_block5.init_weights()
        self.last_conv.apply(init_weights)
        # self.fc.apply(init_weights)
