import torch.nn as nn
import functools


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_norm_layer(norm):
    if norm == 'none':
        return nn.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class ConvBlock(nn.Module):
    """
    Basic block of our model, includes Sequential model of conv layer -> batch normalization -> Relu activation
    This block keeps the original dimension
    """
    def __init__(self, channels, num_filters, filter_size, activation='relu', stride=1, padding=1, norm='batch_norm'):
        super(ConvBlock, self).__init__()
        Norm = get_norm_layer(norm)
        self.block = nn.Sequential(nn.Conv2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride),
                                   Norm(num_filters),
                                   nn.LeakyReLU(0.2, inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs

    def init_weights(self):
        self.block.apply(init_weights)


class Discriminator(nn.Module):
    def __init__(self, norm):
        super().__init__()
        # 256x256 -> 128x128
        self.first_conv = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        # 128x128 -> 64x64
        self.conv_block1 = ConvBlock(64, 64, 4, stride=2, padding=1, norm=norm)
        # 64x64 -> 32x32
        self.conv_block2 = ConvBlock(64, 128, 4, stride=2, padding=1, norm=norm)
        # 32x32 -> 16x16
        self.conv_block3 = ConvBlock(128, 256, 4, stride=2, padding=1, norm=norm)
        # 16x16 -> 8x8
        self.conv_block4 = ConvBlock(256, 512, 4, stride=2, padding=1, norm=norm)
        # 8x8 -> 4x4
        self.conv_block5 = ConvBlock(512, 512, 4, stride=2, padding=1, norm=norm)
        # 4x4 -> 1x1
        self.last_conv = nn.Conv2d(512, 1, 4, stride=1, padding=0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.leaky_relu(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.last_conv(x)
        return x

    def init_weights(self):
        self.first_conv.apply(init_weights)
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.conv_block3.init_weights()
        self.conv_block4.init_weights()
        self.conv_block5.init_weights()
        self.last_conv.apply(init_weights)
