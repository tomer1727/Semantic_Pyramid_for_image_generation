import torch
from torch.autograd import Variable
from PIL import Image
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from classifier import Classifier
from generator2 import Generator


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


def print_images_with_output(image, output, model_name):
    """
    Show @param image, output side by side, used to show input image with its output
    """
    image = np.transpose(image, (1, 2, 0))
    output = np.transpose(output, (1, 2, 0))
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(output)
    plt.axis('off')
    plt.suptitle(model_name)
    plt.show()


def _main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # train_root = r"C:\Users\tomer\OneDrive\Desktop\u"
    # train_root = '/home/dcor/datasets/smallPlaces'
    train_root = '/home/dcor/ronmokady/workshop21/team1/u'

    image_size = 256
    cropped_image_size = 224
    print("set image folder")
    train_set = ImageFolderWithPaths(root=train_root,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(cropped_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]))
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(cropped_image_size)])

    # classifier = torch.load(r"C:\Users\tomer\OneDrive\Desktop\sadna\pyramid_project\classifier")
    classifier = torch.load('./classifier18')
    classifier.eval()
    generator = Generator()
    if args.full_model_name is not None:
        generator.load_state_dict(torch.load('./' + args.full_model_name))
    else:
        generator.load_state_dict(torch.load('./' + args.model_name + '/' + args.model_name + 'G_' + args.epochs))
    # generator.load_state_dict(torch.load('./tanh16/tanh16_200'))

    generator.eval()
    classifier.to(device)
    generator.to(device)

    noise = torch.zeros(1, 256, 1, 1, device=torch.device('cuda:0'))

    print("Starting Eval Loop...")
    train_loss = 0.0  # monitor training loss
    i = 0
    for data in train_loader:
        images, _, paths = data
        images = images.cuda()  # change to gpu tensor
        _, features = classifier(images)
        # features = features[1:5] # for now working with res blocks only
        outputs_images = generator(noise, features)  # forward pass
        outputs_images = 0.5 * (outputs_images + 1)
        image_to_show = outputs_images[0]
        # image_to_show = image_to_show.detach().cpu()
        # print_images_with_output(images[0].detach().cpu().numpy(), image_to_show.numpy(), 'hi')
        in_image = Image.open(paths[0])
        in_image = loader(in_image)
        # image_to_save = image_to_show
        in_image.save('./output_images3/in{}.jpg'.format(i))
        save_image(image_to_show.detach().cpu(), './output_images3/out{}.jpg'.format(i))
        # image_to_save.save('./output_images2/out{}.jpg'.format(i))
        i += 1
        if i == 10:
            # print(np.array(image_to_save))
            # print(image_to_show.numpy())
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--epochs', default='200')
    parser.add_argument('--model-name', default='firstTry')
    parser.add_argument('--full-model-name')
    args = parser.parse_args()
    _main()
