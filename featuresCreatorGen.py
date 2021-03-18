from generator import *
from discriminator import DiscriminatorBlock
import torch


class Features1ToImage(nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64 -> 128x128
        self.conv_block6 = GeneratorBlock(64, 64, 'res')
        # 128x128 -> 256x256
        self.conv_block7 = ResidualBlock(64, 64)
        self.last_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, features1):
        x = self.conv_block6(features1)
        x = self.conv_block7(x)
        x = self.last_conv(x)
        x = self.tanh(x)
        return x


def copy_Features1ToImage_from_gen(generator, output_path):
    f = Features1ToImage()
    f.conv_block6 = generator.conv_block6
    f.conv_block7 = generator.conv_block7
    f.last_conv = generator.last_conv
    torch.save(f.state_dict(),  output_path)


class LevelUpFeaturesGenerator(nn.Module):
    def __init__(self, input_level_features):
        super().__init__()
        level_to_filters_dictionary = {1: 64, 2: 128, 3: 256, 4: 512}
        input_filters = level_to_filters_dictionary[input_level_features]
        output_filters = level_to_filters_dictionary[input_level_features - 1]
        self.res_block1 = ResidualBlock(input_filters, input_filters, need_upsample=True)
        self.res_block2 = ResidualBlock(input_filters, output_filters, need_upsample=False)

    def forward(self, features2):
        x = self.res_block1(features2)
        x = self.res_block2(x)
        return x

    def init_weights(self):
        self.res_block1.init_weights()
        self.res_block2.init_weights()


class FeaturesDiscriminator(nn.Module):
    def __init__(self, norm, dis_type, dis_level):
        super().__init__()
        self.dis_type = dis_type
        self.dis_level = dis_level
        if dis_type == 1:
            # 64x64 -> 32x32
            self.conv_block2 = DiscriminatorBlock(64, 128, dis_type=dis_type, norm=norm)
        if dis_type > 3:
            # 32x32 -> 16x16
            self.conv_block3 = DiscriminatorBlock(128, 256, dis_type=dis_type, norm=norm)
        # 16x16 -> 8x8
        self.conv_block4 = DiscriminatorBlock(256, 512, dis_type=dis_type, norm=norm)
        # 8x8 -> 4x4
        self.conv_block5 = DiscriminatorBlock(512, 512, dis_type=dis_type, norm=norm)
        # 4x4 -> 1x1
        self.last_conv = nn.Conv2d(512, 1, 4, stride=1, padding=0)

    def forward(self, x):
        if self.dis_type == 1:
            x = self.conv_block2(x)
        if self.dis_type > 3:
            x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.last_conv(x)
        return x

    def init_weights(self):
        if self.dis_type == 1:
            self.conv_block2.init_weights()
        if self.dis_type > 3:
            self.conv_block3.init_weights()
        self.conv_block4.init_weights()
        self.conv_block5.init_weights()
        self.last_conv.apply(init_weights)

