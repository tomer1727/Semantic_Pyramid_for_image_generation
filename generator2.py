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


class ConvTransposeBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> Relu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride, bias=False),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs

    def init_weights(self):
        self.block.apply(init_weights)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = ConvTransposeBlock(in_channels, out_channels, 4, stride=2, padding=1)
        self.features_conv_block = ConvTransposeBlock(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, gen, features):
        gen = self.conv_block1(gen)
        features = self.features_conv_block(features)
        return gen + features

    def init_weights(self):
        self.conv_block1.init_weights()
        self.features_conv_block.init_weights()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 -> 4x4
        self.conv_block1 = ConvTransposeBlock(128, 512, 4, stride=1, padding=0)
        # 4x4 -> 8x8
        self.conv_block2 = ConvTransposeBlock(512, 512, 4, stride=2, padding=1)
        # 8x8 -> 16x16
        self.conv_block3 = GeneratorBlock(512, 256)
        # 16x16 -> 32x32
        self.conv_block4 = GeneratorBlock(256, 128)
        # 32x32 -> 64x64
        self.conv_block5 = GeneratorBlock(128, 64)
        # 64x64 -> 128x128
        self.conv_block6 = GeneratorBlock(64, 64)
        # 128x128 -> 256x256
        self.last_conv = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, features):
        # noise has dimension 128 * 1 * 1
        x = self.conv_block1(noise)
        x = self.conv_block2(x)
        x = self.conv_block3(x, features[4])
        x = self.conv_block4(x, features[3])
        x = self.conv_block5(x, features[2])
        x = self.conv_block6(x, features[1])
        x = self.last_conv(x)
        x = self.tanh(x)
        return x

    def init_weights(self):
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.conv_block3.init_weights()
        self.conv_block4.init_weights()
        self.conv_block5.init_weights()
        self.conv_block6.init_weights()
        self.last_conv.apply(init_weights)


