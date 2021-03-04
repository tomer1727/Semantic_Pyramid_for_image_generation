import torch
import torch.nn as nn
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import time
import random
import tensorboardX # run command on terminal: tensorboard --logdir=<your_log_dir>
import functools
import numpy as np
import torchvision.utils as vutils

from classifier import Classifier
from generator2 import Generator
from discriminator2 import Discriminator
from featuresCreatorGen import Features1ToImage, LevelUpFeaturesGenerator


def print_gpu_details():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


def train_generator(features_gen, features_gen_optimizer, features, features_to_train, criterion_next_level_features):
    features_gen.train()

    next_level_created_features = features_gen(features[features_to_train])

    next_level_created_features_loss = criterion_next_level_features(features[features_to_train - 1], next_level_created_features)
    losses_dictionary = {'next_level_loss': next_level_created_features_loss}

    features_gen.zero_grad()
    next_level_created_features_loss.backward()
    features_gen_optimizer.step()

    return losses_dictionary


def sample(f1_to_img, features_generator, features, features_level):
    with torch.no_grad():
        for features_gen in features_generator:
            features_gen.eval()
        f1_to_img.eval()
        features_to_use = features[features_level]
        for i in range(features_level - 2, -1, -1):
            features_to_use = features_generator[i](features_to_use)
        return f1_to_img(features_to_use)


def _main():
    print_gpu_details()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    train_root = args.train_path

    image_size = 256
    cropped_image_size = 256
    print("set image folder")
    train_set = dset.ImageFolder(root=train_root,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(cropped_image_size),
                                     transforms.ToTensor()
                                 ]))

    normalizer_clf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    classifier = torch.load("../try/classifier18")
    classifier.eval()
    classifier.to(device)
    features1_to_image_gen = Features1ToImage()
    features1_to_image_gen.load_state_dict(torch.load('./features_creator_models/features1_to_image'))
    features1_to_image_gen.eval()
    features1_to_image_gen.to(device)
    features_generators = [LevelUpFeaturesGenerator(input_level_features=i) for i in range(2, 5)]
    for i, features_gen in enumerate(features_generators):
        input_level_features = i + 2
        features_gen.to(device)
        # weights init
        features_gen.load_state_dict(torch.load('./features_creator_models/features{}_to_features{}'.format(input_level_features, input_level_features - 1)))
        # features_gen.init_weights()

    # losses + optimizers
    next_level_features_criterions = [nn.MSELoss() for i in range(2, 5)]
    optimizers = [optim.Adam(features_gen.parameters(), lr=args.lr, betas=(0.5, 0.999)) for features_gen in features_generators]

    num_of_epochs = args.epochs

    starting_time = time.time()
    iterations = 0
    outputs_dir = os.path.join('features_creation_models', args.model_name)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
    temp_results_dir = os.path.join(outputs_dir, 'temp_results')
    if not os.path.isdir(temp_results_dir):
        os.mkdir(temp_results_dir)
    models_dir = os.path.join(outputs_dir, 'models_checkpoint')
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    writer = tensorboardX.SummaryWriter(os.path.join(outputs_dir, 'summaries'))
    fixed_features = 0
    first_iter = True
    print("Starting Training Loop...")
    train_features_dictionary = {0: 2, 1: 3, 2: 3, 3: 4, 4: 4, 5: 4}
    for epoch in range(num_of_epochs):
        for data in train_loader:
            iterations += 1
            if iterations % 30 == 1:
                print('epoch:', epoch, ', iter', iterations, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.to(device)  # change to gpu tensor
            images_clf = normalizer_clf(images)
            _, features = classifier(images_clf)
            features = list(features)
            if first_iter:
                first_iter = False
                fixed_features = [torch.clone(features[x]) for x in range(len(features))]
                grid = vutils.make_grid(images, padding=2, normalize=False, nrow=8)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'original_images.jpg'))
            for i, features_gen in enumerate(features_generators):
                features_to_train = i + 2
                if i != train_features_dictionary[iterations % 6]:
                    continue
                generator_loss_dict = train_generator(features_gen, optimizers[i], features, features_to_train, next_level_features_criterions[i])
                for k, v in generator_loss_dict.items():
                    writer.add_scalar('G/f' + str(features_to_train) + '_%s' % k, v.data.cpu().numpy(), global_step=iterations//1 + 1)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))

            if iterations < 10000 and iterations % 2000 == 1 or iterations % 4000 == 1:
                for i, features_gen in enumerate(features_generators):
                    features_level = i + 2
                    torch.save(features_gen.state_dict(),  models_dir + '/' + args.model_name + '_f{}_to_f{}'.format(features_level, features_level - 1))
                    # regular sampling (#batch_size different images)
                    fake_images = sample(features1_to_image_gen, features_generators, fixed_features, features_level)
                    grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=8)
                    vutils.save_image(grid, os.path.join(temp_results_dir, 'res_iter_{}_origin_f{}.jpg'.format(iterations // 2000, features_level)))

            if iterations % 20000 == 1:
                for i, features_gen in enumerate(features_generators):
                    features_level = i + 2
                    torch.save(features_gen.state_dict(), models_dir + '/' + args.model_name + '_f{}_to_f{}_'.format(features_level, features_level - 1) + str(iterations // 20000))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--model-name')
    parser.add_argument('--discriminator-norm', default='instance_norm', choices=['batch_norm', 'instance_norm', 'layer_norm'])
    parser.add_argument('--gradient-penalty-weight', type=float, default=10.0)
    parser.add_argument('--discriminator-steps', type=int, default=5)
    parser.add_argument('--gen-type', default='default', choices=['default', 'res'])
    parser.add_argument('--train-path', default='/home/dcor/datasets/places365')
    args = parser.parse_args()
    print(args)
    if args.model_name is None:
        print("Must specify model name")
        exit(1)
    out_dir = './features_creation_models/' + args.model_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    arg_dic = vars(args)
    with open(out_dir + '/hyperparams.txt', 'w') as file:
        for key in arg_dic:
            file.write(key + ': {}\n'.format(arg_dic[key]))

    _main()
