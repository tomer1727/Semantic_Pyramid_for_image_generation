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
import tensorboardX
import functools
import torchvision.utils as vutils

from classifier import Classifier
from featuresCreatorGen import Features1ToImage, LevelUpFeaturesGenerator, FeaturesDiscriminator
from train_model import get_wgan_losses_fn, gradient_penalty


def print_gpu_details():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


def train_generator(features_gen, discriminator, classifier, features_gen_optimizer, features, features_to_train, generator_loss_fn, criterion_features, next_level_features_criterion):
    features_gen.train()
    discriminator.train()

    next_level_created_features = features_gen(features[features_to_train])

    # wgan-gp loss
    fake_images_d_logit = discriminator(next_level_created_features)
    generator_loss = generator_loss_fn(fake_images_d_logit)

    # features loss
    if features_to_train == 2:
        reconstruct_features = classifier.layer2(next_level_created_features)
    elif features_to_train == 3:
        reconstruct_features = classifier.layer3(next_level_created_features)
    else: # level 4
        reconstruct_features = classifier.layer4(next_level_created_features)
    features_loss = criterion_features(features[features_to_train], reconstruct_features)

    # next level loss
    next_level_features_loss = next_level_features_criterion(next_level_created_features, features[features_to_train - 1])

    losses_dictionary = {'generator_loss': generator_loss, 'features_loss': features_loss, 'next_level_loss': next_level_features_loss}
    total_loss = generator_loss + features_loss + 0.2 * next_level_features_loss
    features_gen.zero_grad()
    total_loss.backward()
    features_gen_optimizer.step()

    return losses_dictionary


def train_discriminator(features_gen, discriminator, discriminator_loss_fn, discriminator_optimizer, features, features_to_train):
    features_gen.train()
    discriminator.train()

    fake_features = features_gen(features[features_to_train]).detach()

    real_features_d_logit = discriminator(features[features_to_train - 1])
    fake_features_d_logit = discriminator(fake_features)

    real_features_d_logit, fake_features_d_logit = discriminator_loss_fn(real_features_d_logit, fake_features_d_logit)
    gp = gradient_penalty(functools.partial(discriminator), features[features_to_train - 1], fake_features)

    discriminator_loss = (real_features_d_logit + fake_features_d_logit) + gp * args.gradient_penalty_weight

    discriminator.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()

    return {'d_loss': real_features_d_logit + fake_features_d_logit, 'gp': gp}


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

    classifier = torch.load(args.classifier_path)
    classifier.eval()
    classifier.to(device)
    features1_to_image_gen = Features1ToImage()
    features1_to_image_gen.load_state_dict(torch.load(os.path.join(args.features_gens_dir_path, 'features1_to_image')))
    features1_to_image_gen.eval()
    features1_to_image_gen.to(device)
    features_generators = [LevelUpFeaturesGenerator(input_level_features=i) for i in range(2, 4)]
    for i, features_gen in enumerate(features_generators):
        input_level_features = i + 2
        features_gen.to(device)
        # weights init
        if input_level_features < args.train_block_input:
            features_gen.load_state_dict(torch.load(os.path.join(args.features_gens_dir_path, 'features{}_to_features{}'.format(input_level_features, input_level_features - 1))))
            features_gen.eval()
        else:
            features_gen.init_weights()

    discriminator = FeaturesDiscriminator(args.discriminator_norm, dis_type=args.gen_type, dis_level=3)
    discriminator.to(device)
    discriminator.init_weights()

    # losses + optimizers
    criterion_discriminator, criterion_generator = get_wgan_losses_fn()
    next_level_features_criterion = nn.L1Loss()
    criterion_features = nn.L1Loss()
    gen_optimizer = optim.Adam(features_generators[args.train_block_input - 2].parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

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
    features_to_train = args.train_block_input
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

            if iterations % (args.discriminator_steps + 1) != 1:
                discriminator_loss_dict = train_discriminator(features_generators[features_to_train - 2], discriminator, criterion_discriminator, discriminator_optimizer, features, features_to_train)
                for k, v in discriminator_loss_dict.items():
                    writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=iterations)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))
            else:
                generator_loss_dict = train_generator(features_generators[features_to_train - 2], discriminator, classifier, gen_optimizer, features,
                                                      features_to_train, criterion_generator, criterion_features, next_level_features_criterion)
                for k, v in generator_loss_dict.items():
                    writer.add_scalar('G/f' + str(features_to_train) + '_%s' % k, v.data.cpu().numpy(), global_step=iterations//1 + 1)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))

            if iterations < 10000 and iterations % 2000 == 1 or iterations % 4000 == 1:
                for i, features_gen in enumerate(features_generators):
                    features_level = i + 2
                    if features_level == features_to_train: # print only the given layer output (have option to generate from all layer by modifying the if)
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
    parser.add_argument('--gen-type', default='res', choices=['default', 'res'])
    parser.add_argument('--train-path', default='/home/dcor/datasets/places365')
    parser.add_argument('--classifier-path', default='/home/dcor/ronmokady/workshop21/team1/try/classifier18')
    parser.add_argument('--features-gens-dir-path', help='the dir that the pre trained generators are located at', required=True)
    parser.add_argument('--train-block-input', default=3, choices=[2, 3, 4], help='train the block that creates features from the given level')
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
