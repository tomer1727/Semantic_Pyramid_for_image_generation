import torch
from torch.autograd import Variable
from PIL import Image
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import argparse
import time
import random

from classifier import Classifier
from generator2 import Generator
from discriminator2 import Discriminator


real_label_D = 0.9
real_label_G = 1.0
fake_label = 0.1


def print_gpu_details():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)


def _main():
    print_gpu_details()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    train_root = args.train_path
    # train_root = r"C:\Users\tomer\OneDrive\Documents\datasets\model2\cars\train"

    image_size = 256
    cropped_image_size = 224
    print("set image folder")
    train_set = dset.ImageFolder(root=train_root,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(cropped_image_size),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]))
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # classifier = torch.load(r"C:\Users\tomer\OneDrive\Desktop\sadna\pyramid_project\classifier")
    classifier = torch.load("./classifier18")
    classifier.eval()
    generator = Generator()
    # generator.load_state_dict(torch.load('./full_fe2/full_fe2G'))
    discriminator = Discriminator()
    # discriminator.load_state_dict(torch.load('./expc/expcD'))
    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    classifier.to(device)
    generator.to(device)
    discriminator.to(device)

    # weights init
    generator.init_weights()
    discriminator.init_weights()

    # losses + optimizers
    criterion_features = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.99))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=2*args.lr, betas=(0.5, 0.99))

    num_of_epochs = args.epochs

    normalizer_clf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalizer_discriminator = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # noise = torch.zeros(args.batch_size, 512, 1, 1, device=torch.device('cuda:0'))
    starting_time = time.time()
    iterations = 0
    temp_results_dir = os.path.join(args.model_name, 'temp_results')
    if not os.path.isdir(temp_results_dir):
        os.mkdir(temp_results_dir)
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        for data in train_loader:
            iterations += 1
            if iterations % 100 == 1:
                print('epoch:', epoch, ', iter', iterations, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.to(device)  # change to gpu tensor
            images_clf = normalizer_clf(images)

            # generator update
            # optimizer_generator.zero_grad()  # zeros previous grads
            generator.zero_grad()
            _, features = classifier(images_clf)
            # if images.shape[0] != noise.shape[0]:
            #     del noise
            # (torch.randn(images.shape[0], 3, 224, 224, device=device)) * std
            noise = torch.randn(images.shape[0], 256, 1, 1, device=device)
            features_to_train = 2
            features = list(features)
            for i in range(len(features)):
                if i != features_to_train:
                    features[i] = features[i] * 0
            if iterations == 1:
                fixed_features = [x.clone() for x in features]
                fixed_noise = noise.clone()
            fake_images = generator(noise, features)
            fake_images = 0.5 * (fake_images + 1)
            fake_images_normalized = normalizer_clf(fake_images)

            # if iterations % 4 != 2:
            # print('Generator update')
            label = torch.full((images.shape[0],), real_label_G, dtype=torch.float, device=device)
            discriminator_preds = discriminator(fake_images).view(-1)
            loss_adversarial = criterion_bce(discriminator_preds, label)
            # loss_adversarial.backward()
            total_loss = loss_adversarial
            # optimizer_generator.step()  # modify weights
            _, outputs_images_features = classifier(fake_images_normalized)
            loss_features = criterion_features(features[features_to_train], outputs_images_features[features_to_train])
            # for i in range(1, len(features) - 1):
            #     loss_features += criterion_features(features[i], outputs_images_features[i])  # calculate loss
            # loss_features += criterion_features(images, fake_images)
            # loss_features.backward()
            if iterations % 20 == 2:
                print('features to train: {}, features loss: {:.6f}'.format(features_to_train, loss_features.item()))
            total_loss += 5 * loss_features
            del outputs_images_features

            total_loss.backward()
            optimizer_generator.step()  # modify weights

            if iterations % 2 == 1:
            # if True:
                # discriminator update
                # print('Discriminator update')
                # optimizer_discriminator.zero_grad()
                discriminator.zero_grad()
                images = normalizer_discriminator(images)
                # real images batch
                label = torch.full((images.shape[0],), real_label_D, dtype=torch.float, device=device)
                label += (torch.randn(images.shape[0], device=device)) * 0.04
                flip = torch.rand(1)[0] > 1
                flip = flip.item()
                if flip:
                    label[random.randint(0, images.shape[0] - 1)] = fake_label
                std = 0.1 / (iterations // 1000 + 1)
                white_noise = (torch.randn(images.shape[0], 3, 224, 224, device=device)) * std
                images = images + white_noise
                output = discriminator(images).view(-1) # forward pass
                acc = torch.sum(output > 0.5)
                loss_real = criterion_bce(output, label)
                loss_real.backward()
                # fake batch
                label.fill_(fake_label)
                label += (torch.randn(images.shape[0], device=device)) * 0.04
                label = torch.max(torch.zeros(images.shape[0], device=device), label)
                if flip:
                    label[random.randint(0, images.shape[0] - 1)] = real_label_D
                fake_images = fake_images + white_noise
                output = discriminator(fake_images.detach()).view(-1) # forward pass
                acc += torch.sum(output < 0.5)
                acc_np = acc.cpu().numpy()
                loss_fake = criterion_bce(output, label)
                loss_fake.backward()
                discriminator_loss = loss_real + loss_fake
                # discriminator_loss.backward()
                optimizer_discriminator.step()
                if iterations % 6 == 1:
                    print('discriminator accuracy:', acc_np, '/', args.batch_size * 2)
                del white_noise
                del loss_fake
                del loss_real
                del acc

            if iterations % 30 == 1:
                # print('iter: {} \tfeatures Loss: {:.6f}, discriminator loss: {:.6f}, generator loss: {:.6f}'.format(
                #     batch_count, loss_features.item(), discriminator_loss.item(), loss_adversarial.item()))
                print('iter: {} \t, discriminator loss: {:.6f}, generator loss: {:.6f}'.format(iterations, discriminator_loss.item(), loss_adversarial.item()))
                # print('iter: {} \tfeatures Loss: {:.6f}'.format(batch_count, loss_features.item()))
                del loss_adversarial
            del data
            del features
            del images
            del fake_images
            del images_clf
            del noise
            if iterations % 3 == 1:
                del discriminator_loss

            if iterations % 1000 == 1:
                torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + 'G')
                torch.save(discriminator.state_dict(), './' + args.model_name + '/' + args.model_name + 'D')
                with torch.no_grad():
                    fake = generator(fixed_noise, fixed_features).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'res_iter_{}.jpg'.format(iterations // 1000)))
            if iterations % 15000 == 1:
                torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + 'G_' + str(iterations // 15000))
                torch.save(discriminator.state_dict(), './' + args.model_name + '/' + args.model_name + 'D_' + str(iterations // 15000))

        # print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch + 1, num_of_epochs, train_loss))
        # save the model, if needed
        if (not args.only_final and epoch % 10 == 0) or epoch == num_of_epochs - 1:
            torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + 'G_' + str(epoch + 1))
            torch.save(discriminator.state_dict(), './' + args.model_name + '/' + args.model_name + 'D_' + str(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
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
