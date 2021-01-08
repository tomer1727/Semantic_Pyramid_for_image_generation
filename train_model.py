import torch
from torch.autograd import Variable
from PIL import Image
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import argparse
import time

from classifier import Classifier
from generator import Generator
from discriminator import Discriminator


real_label = 1.
fake_label = 0.

def PrintGpuDetails():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


def WeightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Block') != -1:
        # m.apply(WeightsInit)
        return
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def LoadTrainData(train_root):
    image_size = 256
    cropped_image_size = 224
    print("Setting image folder...")
    train_set = dset.ImageFolder(root=train_root,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(cropped_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                 ]))
    print("Setting data loader...")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return train_loader

def _main():
    PrintGpuDetails()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    train_root = args.train_path
    num_of_epochs = args.epochs
    # train_root = r"C:\Users\tomer\OneDrive\Documents\datasets\model2\cars\train"

    train_data = LoadTrainData(train_root)

    # classifier = torch.load(r"C:\Users\tomer\OneDrive\Desktop\sadna\pyramid_project\classifier")
    classifier = torch.load("./classifier")
    classifier.eval()
    generator = Generator()
    discriminator = Discriminator()

    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    classifier.to(device)
    generator.to(device)
    discriminator.to(device)

    # Initialize weights
    generator.apply(WeightsInit)
    discriminator.apply(WeightsInit)

    # Define loss functions and optimizers
    criterion_features = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr)

    normalizer = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    noise = torch.zeros(args.batch_size, 2048, 7, 7, device=torch.device('cuda:0'))

    starting_time = time.time()
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        batch_count = 0
        for data in train_data:
            batch_count += 1
            if batch_count % 10 == 1:
                print('epoch:', epoch, ', batch', batch_count, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.cuda()  # change to gpu tensor
            # discriminator update
            print('Training step of Discriminator')
            optimizer_discriminator.zero_grad()
            discriminator.zero_grad()
            # real images batch
            label = torch.full((images.shape[0],), real_label, dtype=torch.float, device=device)
            output = discriminator(images).view(-1) # forward pass
            loss_real = criterion_bce(output, label)
            loss_real.backward()
            # fake batch
            _, features = classifier(images)
            features = features[1:5]  # for now working with res blocks only
            if images.shape[0] != noise.shape[0]:
                del noise
                noise = torch.zeros(images.shape[0], 2048, 7, 7, device=torch.device('cuda:0'))
            fake_images = generator(noise, features)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1) # forward pass
            loss_fake = criterion_bce(output, label)
            loss_fake.backward()
            discriminator_loss = loss_real + loss_fake
            optimizer_discriminator.step()
            # generator update
            print('Training step of Generator')
            optimizer_generator.zero_grad()  # zeros previous grads
            generator.zero_grad()
            label.fill_(real_label)
            discriminator_preds = discriminator(fake_images).view(-1)
            loss_adversarial = criterion_bce(discriminator_preds, label)
                # loss_adversarial.backward()
            fake_images = normalizer(fake_images)
            _, outputs_images_features = classifier(fake_images)
            outputs_images_features = outputs_images_features[1:5]
            loss_features = criterion_features(features[0], outputs_images_features[0])
            for i in range(1, len(features)):
                loss_features += criterion_features(features[i], outputs_images_features[i])  # calculate loss
            # total_loss = loss_adversarial + loss_features
            # total_loss.backward()  # back prop
            optimizer_generator.step()  # modify weights

            if batch_count % 10 == 1:
                print('iter: {} \tfeatures Loss: {:.6f}, discriminator loss: {:.6f}, generator loss: {:.6f}'.format(
                    batch_count, loss_features.item(), discriminator_loss.item(), loss_adversarial.item()))
            del data
            del features
            del images
            del fake_images
            del outputs_images_features
            del loss_features
            del loss_adversarial
            # del total_loss
            del discriminator_loss
        # print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch + 1, num_of_epochs, train_loss))
        # save the model, if needed
        if (not args.only_final and epoch % 10 == 0) or epoch == num_of_epochs - 1:
            torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + '_' + str(epoch + 1))
        elif epoch == 3 or epoch == 5:
            torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + '_' + str(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--only-final', action='store_true')
    parser.add_argument('--model-name')
    parser.add_argument('--train-path', default='/home/dcor/datasets/places365')
    parser.add_argument('--test-path', default='../testImages')
    args = parser.parse_args()
    print(args)
    if args.model_name is None:
        print("Must specify model name")
        exit(1)
    if not os.path.exists('./' + args.model_name):
        os.mkdir('./' + args.model_name)
    arg_dic = vars(args)
    with open('./' + args.model_name + '/hyperparams.txt', 'w') as file:
        for key in arg_dic:
            file.write(key + ': {}\n'.format(arg_dic[key]))

    _main()
