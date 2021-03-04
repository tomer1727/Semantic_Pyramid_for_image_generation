from generator2 import *
import torch
import os


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


def copy_Features1ToImage_from_gen():
    generator = Generator(gen_type='res')
    # generator.load_state_dict(torch.load('./wgan-gp_models/wgan_level1_content_level2_next_loss/models_checkpoint/expc/wgan_level1_content_level2_next_lossG'))
    generator.load_state_dict(torch.load(r"C:\Users\tomer\OneDrive\Desktop\wgan_level1_content_level2_next_lossG"))
    f = Features1ToImage()
    f.conv_block6 = generator.conv_block6
    f.conv_block7 = generator.conv_block7
    f.last_conv = generator.last_conv
    torch.save(f.state_dict(),  r"C:\Users\tomer\OneDrive\Desktop\features1_to_image")


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


def build_generator_from_features_gen():
    features_generators_dir = r"C:\Users\tomer\OneDrive\Desktop\sadna\features_generators"
    features_generators = [Features1ToImage()] + [LevelUpFeaturesGenerator(input_level_features=i) for i in range(2, 5)]
    for i, features_gen in enumerate(features_generators):
        input_level_features = i + 1
        features_gen.load_state_dict(torch.load(os.path.join(features_generators_dir, "f{}_to_f{}".format(input_level_features, input_level_features - 1))))
    generator = Generator(gen_type='double_res')
    # copy features1_to_img part
    generator.conv_block6 = features_generators[0].conv_block6
    generator.conv_block7 = features_generators[0].conv_block7
    generator.last_conv = features_generators[0].last_conv
    generator_blocks = [generator.conv_block5, generator.conv_block4, generator.conv_block3] # the block that get the features as input
    for i, gen_block in enumerate(generator_blocks):
        generator_blocks[i].conv_block1 = features_generators[i + 1].res_block1 # i + 1 to skip f1_to_img gen
        generator_blocks[i].conv_block2 = features_generators[i + 1].res_block2
    torch.save(generator.state_dict(), os.path.join(features_generators_dir, 'full_generator'))
