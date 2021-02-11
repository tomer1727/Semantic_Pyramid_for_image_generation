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
    def __init__(self, channels, num_filters, filter_size, activation='relu', conv_type='deconv', stride=1, padding=1):
        super(ConvBlock, self).__init__()
        if conv_type == 'deconv':
            conv_layer = nn.ConvTranspose2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride, bias=False)
        else:
            conv_layer = nn.Conv2d(channels, num_filters, kernel_size=filter_size, padding=padding, stride=stride, bias=False)

        self.block = nn.Sequential(conv_layer,
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity())

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs

    def init_weights(self):
        self.block.apply(init_weights)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_block1 = ConvBlock(in_channels, out_channels, 3, conv_type='conv')
        self.conv_block2 = ConvBlock(out_channels, out_channels, 3, conv_type='conv', activation='none')
        self.skip_block = ConvBlock(in_channels, out_channels, 1, stride=1, padding=0, conv_type='conv', activation='none')
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        residual = self.skip_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x += residual
        x = self.activation(x)
        return x

    def init_weights(self):
        self.conv_block1.init_weights()
        self.conv_block2.init_weights()
        self.skip_block.init_weights()


class GeneratorBlock(nn.Module):
    # gen_type param is the generator param, default for the regular implementation, res for resnet implementation
    def __init__(self, in_channels, out_channels, gen_type='default'):
        super().__init__()
        self.gen_type = gen_type
        if gen_type == 'default':
            self.conv_block1 = ConvBlock(in_channels, out_channels, 4, stride=2, padding=1)
            self.features_conv_block = ConvBlock(in_channels, out_channels, 4, stride=2, padding=1)
        else: # resnet generator
            self.conv_block1 = ResidualBlock(in_channels, out_channels)
            self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
            self.features_conv_block = ConvBlock(in_channels, out_channels, 3, conv_type='conv')

    def forward(self, gen, features):
        gen = self.conv_block1(gen)
        if self.gen_type == 'res':
            features = self.upsample(features)
        features = self.features_conv_block(features)
        return gen + features

    def init_weights(self):
        self.conv_block1.init_weights()
        self.features_conv_block.init_weights()


class Generator(nn.Module):
    def __init__(self, gen_type):
        super().__init__()
        self.gen_type = gen_type
        # 1x1 -> 4x4
        self.conv_block1 = ConvBlock(128, 512, 4, stride=1, padding=0)
        # 4x4 -> 8x8
        self.conv_block2 = ConvBlock(512, 512, 4, stride=2, padding=1)
        # 8x8 -> 16x16
        self.conv_block3 = GeneratorBlock(512, 256, gen_type)
        # 16x16 -> 32x32
        self.conv_block4 = GeneratorBlock(256, 128, gen_type)
        # 32x32 -> 64x64
        self.conv_block5 = GeneratorBlock(128, 64, gen_type)
        # 64x64 -> 128x128
        self.conv_block6 = GeneratorBlock(64, 64, gen_type)
        if gen_type == 'default':
            # 128x128 -> 256x256
            self.last_conv = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        else: # resnet generator
            # 128x128 -> 256x256
            self.conv_block7 = ResidualBlock(64, 64)
            self.last_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, features):
        # noise has dimension 128 * 1 * 1
        x = self.conv_block1(noise)
        x = self.conv_block2(x)
        x = self.conv_block3(x, features[4])
        x = self.conv_block4(x, features[3])
        x = self.conv_block5(x, features[2])
        x = self.conv_block6(x, features[1])
        if self.gen_type == 'res':
            x = self.conv_block7(x)
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
        if self.gen_type == 'res':
            self.conv_block7.init_weights()
        self.last_conv.apply(init_weights)


