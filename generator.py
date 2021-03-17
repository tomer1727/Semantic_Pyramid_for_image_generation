import torch
import torch.nn as nn


class ConvTransposeBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> Relu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(nn.utils.spectral_norm(nn.ConvTranspose2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride)),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv_block1 = ConvTransposeBlock(in_channels, in_channels, 1, padding=0)
        self.conv_block2 = ConvTransposeBlock(in_channels, in_channels, 4, stride=2)
        self.conv_block3 = ConvTransposeBlock(in_channels, out_channels, 1, padding=0, activation='none')
        self.skip_block = nn.Sequential(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False)),
                                        nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU()

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


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.features_conv_block = ConvTransposeBlock(in_channels, out_channels, 4, stride=2)

    def forward(self, gen, features):
        gen = self.res_block(gen)
        features = self.features_conv_block(features)
        return gen + features


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_block1 = GeneratorBlock(2048, 1024)
        self.gen_block2 = GeneratorBlock(1024, 512)
        self.gen_block3 = GeneratorBlock(512, 256)
        # self.attn1 = SelfAttentionLayer(in_dim=256)
        self.gen_block4 = GeneratorBlock(256, 64)
        self.last_conv = nn.utils.spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=8, padding=3, stride=2))
        self.conv_block1 = ConvTransposeBlock(3, 3, 3, stride=1, padding=1)
        self.conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(3, 3, kernel_size=3, padding=1, stride=1))
        # self.conv_block2 = ConvTransposeBlock(3, 3, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, features):
        # noise has dimension 2048 * 7 * 7 (TODO: check adding layer here)
        x = self.gen_block1(noise, features[3])
        x = self.gen_block2(x, features[2])
        x = self.gen_block3(x, features[1])
        x, _ = self.attn1(x)
        x = self.gen_block4(x, features[0])
        x = self.last_conv(x)
        x = self.conv_block1(x)
        x = self.conv2(x)
        x = self.tanh(x)
        return x
