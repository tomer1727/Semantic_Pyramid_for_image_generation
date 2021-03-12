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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch_norm'):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv_block1 = ConvBlock(in_channels, in_channels, 3, norm=norm)
        self.conv_block2 = ConvBlock(in_channels, out_channels, 3, norm=norm, activation='none')
        self.skip_block = ConvBlock(in_channels, out_channels, 1, stride=1, padding=0, norm=norm, activation='none')
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        residual = self.skip_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x += residual
        x = self.activation(x)
        x = self.downsample(x)
        return x

    def init_weights(self):
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.skip_block.init_weights()


class DiscriminatorBlock(nn.Module):
    # dis_type param is the generator param, default for the regular implementation, res for resnet implementation
    def __init__(self, in_channels, out_channels, dis_type='default', norm='batch_norm'):
        super().__init__()
        self.dis_type = dis_type
        if dis_type == 'default':
            self.conv_block1 = ConvBlock(in_channels, out_channels, 4, stride=2, padding=1, norm=norm)
        else: # resnet discriminator
            self.conv_block1 = ResidualBlock(in_channels, out_channels, norm)

    def forward(self, dis):
        dis = self.conv_block1(dis)
        return dis

    def init_weights(self):
        self.conv_block1.init_weights()


class Discriminator(nn.Module):
    def __init__(self, norm, dis_type):
        super().__init__()
        self.dis_type = dis_type
        # 256x256 -> 128x128
        if dis_type == 'default':
            self.first_conv = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
            self.leaky_relu = nn.LeakyReLU(0.2, True)
        else:
            self.first_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.leaky_relu = nn.LeakyReLU(0.2, True)
            self.avg_pool = nn.AvgPool2d(2)
        # 128x128 -> 64x64
        self.conv_block1 = DiscriminatorBlock(64, 64, dis_type=dis_type, norm=norm)
        # 64x64 -> 32x32
        self.conv_block2 = DiscriminatorBlock(64, 128, dis_type=dis_type, norm=norm)
        # 32x32 -> 16x16
        self.conv_block3 = DiscriminatorBlock(128, 256, dis_type=dis_type, norm=norm)
        # 16x16 -> 8x8
        self.conv_block4 = DiscriminatorBlock(256, 512, dis_type=dis_type, norm=norm)
        # 8x8 -> 4x4
        self.conv_block5 = DiscriminatorBlock(512, 512, dis_type=dis_type, norm=norm)
        # 4x4 -> 1x1
        self.last_conv = nn.Conv2d(512, 1, 4, stride=1, padding=0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.leaky_relu(x)
        if self.dis_type == 'res':
            x = self.avg_pool(x)
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
