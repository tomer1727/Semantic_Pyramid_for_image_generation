from generator2 import *
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


def copy_Features1ToImage_from_gen():
    generator = Generator(gen_type='res')
    # generator.load_state_dict(torch.load('./wgan-gp_models/wgan_level1_content_level2_next_loss/models_checkpoint/expc/wgan_level1_content_level2_next_lossG'))
    generator.load_state_dict(torch.load(r"C:\Users\tomer\OneDrive\Desktop\wgan_level1_content_level2_next_lossG"))
    f = Features1ToImage()
    f.conv_block6 = generator.conv_block6
    f.conv_block7 = generator.conv_block7
    f.last_conv = generator.last_conv
    torch.save(f.state_dict(),  r"C:\Users\tomer\OneDrive\Desktop\features1_to_image")


class Features2ToFeatures1(nn.Module):
    def __init__(self):
        super().__init__()
        # 32x32 -> 64x64
        self.res_block1 = ResidualBlock(128, 128, need_upsample=True)
        self.res_block2 = ResidualBlock(128, 64, need_upsample=False)

    def forward(self, features2):
        x = self.res_block1(features2)
        x = self.res_block2(x)
        return x

    def init_weights(self):
        self.res_block1.init_weights()
        self.res_block2.init_weights()
