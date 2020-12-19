import torch.nn as nn


class ConvTransposeBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> leakyRelu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride),
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
        self.skip_block = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False),
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
        self.gen_block4 = GeneratorBlock(256, 64)
        self.last_conv_block = ConvTransposeBlock(64, 3, 8, stride=2, padding=3)

    def forward(self, noise, features):
        # noise has dimension 2048 * 7 * 7 (TODO: check adding layer here)
        x = self.gen_block1(noise, features[3])
        x = self.gen_block2(x, features[2])
        x = self.gen_block3(x, features[1])
        x = self.gen_block4(x, features[0])
        x = self.last_conv_block(x)
        return x
