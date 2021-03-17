import torch
from PIL import Image
import os
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import argparse

from classifier import Classifier
from generator import Generator


class ImageFolderWithPaths(dset.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def _main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator = Generator()
    generator.load_state_dict(torch.load(args.full_model_name))

    generator.eval()
    generator.to(device)

    print("Starting Eval Loop...")
    output_images_dir = os.path.join(args.output, args.full_model_name.split('/')[-1])
    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir, exist_ok=True)
    for j in range(15):
        z = torch.randn(1, 128, 1, 1, device=device)
        with torch.no_grad():
            generator.eval()
            outputs_images = generator(z)
            outputs_images = 0.5 * (outputs_images + 1)
            save_image(outputs_images[0].detach().cpu(), os.path.join(output_images_dir, 'out_{}.jpg'.format(j)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--full-model-name', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    _main()
