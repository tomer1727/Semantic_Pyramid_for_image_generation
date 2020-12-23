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


def print_gpu_details():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


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
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]))
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # classifier = torch.load(r"C:\Users\tomer\OneDrive\Desktop\sadna\pyramid_project\classifier")
    classifier = torch.load("./classifier")
    classifier.eval()
    generator = Generator()
    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
        generator = nn.DataParallel(generator)
    classifier.to(device)
    generator.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)

    num_of_epochs = args.epochs

    normalizer = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    noise = torch.zeros(args.batch_size, 2048, 7, 7, device=torch.device('cuda:0'))
    starting_time = time.time()
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        print('epoch:', epoch)
        train_loss = 0.0  # monitor training loss
        batch_count = 0
        for data in train_loader:
            batch_count += 1
            if batch_count % 10 == 1:
                print('batch', batch_count, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.cuda()  # change to gpu tensor
            optimizer.zero_grad()  # zeros previous grads
            _, features = classifier(images)
            features = features[1:5] # for now working with res blocks only
            if images.shape[0] != noise.shape[0]:
                del noise
                noise = torch.zeros(images.shape[0], 2048, 7, 7, device=torch.device('cuda:0'))
            outputs_images = generator(noise, features)  # forward pass
            outputs_images = normalizer(outputs_images)
            _, outputs_images_features = classifier(outputs_images)
            outputs_images_features = outputs_images_features[1:5]
            loss = criterion(features[0], outputs_images_features[0])
            for i in range(1, len(features)):
                loss += criterion(features[i], outputs_images_features[i])  # calculate loss
            loss.backward()  # back prop
            optimizer.step()  # modify weights
            train_loss += loss.item() * images.size(0)
            del data
            del features
            del images
            del outputs_images
            del outputs_images_features
            del loss
        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch + 1, num_of_epochs, train_loss))
        # save the model, if needed
        if (not args.only_final and epoch % 10 == 0) or epoch == num_of_epochs - 1:
            torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + '_' + str(epoch + 1))
        elif epoch == 3 or epoch == 5:
            torch.save(generator.state_dict(), './' + args.model_name + '/' + args.model_name + '_' + str(epoch + 1))


    # test_root = args.test_path
    # test_set = dset.ImageFolder(root=test_root,
    #                             transform=transforms.Compose([
    #                                 transforms.Resize(image_size),
    #                                 transforms.CenterCrop(image_size),
    #                                 transforms.ToTensor(),
    #                             ]))


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
