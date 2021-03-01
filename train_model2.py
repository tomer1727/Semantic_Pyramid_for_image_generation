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


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = -r_logit.mean()
        f_loss = f_logit.mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def sample_line(real, fake):
    shape = [real.size(0)] + [1] * (real.dim() - 1)
    alpha = torch.rand(shape, device=real.device)
    sample = real + alpha * (fake - real)
    return sample


def norm(x):
    normal = x.view(x.size(0), -1).norm(p=2, dim=1)
    return normal


def one_mean_gp(grad):
    normal = norm(grad)
    gp = ((normal - 1)**2).mean()
    return gp


def gradient_penalty(f, real, fake):
    x = sample_line(real, fake).detach()
    x.requires_grad = True
    pred = f(x)
    grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    gp = one_mean_gp(grad)

    return gp


def train_generator(generator, discriminator, generator_loss_fn, generator_optimizer, batch_size, features, criterion_features, features_to_train,
                    classifier, normalizer_clf, criterion_diversity_n, criterion_diversity_d, criterion_next_level_features, epsilon=10e-4):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(batch_size, 128, 1, 1, device=device)

    fake_images, next_level_created_features = generator(z, features, features_to_train)
    # wgan-gp loss
    fake_images_d_logit = discriminator(fake_images)
    generator_loss = generator_loss_fn(fake_images_d_logit)

    # content loss (reconstruction loss)
    fake_images_clf = normalizer_clf(fake_images)
    _, fake_features = classifier(fake_images_clf)
    need_init = True
    content_loss = 0
    for i in range(1, 4 + 1):
        # normalize_factor = features[i].shape[1] * features[i].shape[2] * features[i].shape[3]
        normalize_factor = 1
        if need_init:
            need_init = False
            content_loss = (1/normalize_factor) * criterion_features(features[i], fake_features[i])
        else:
            content_loss += (1/normalize_factor) * criterion_features(features[i], fake_features[i])

    # diversity loss
    # z2 = torch.randn(batch_size, 128, 1, 1, device=device)
    # fake_images2 = generator(z2, features)
    # diversity_loss = criterion_diversity_n(z, z2) / (criterion_diversity_d(fake_images, fake_images2) + epsilon)

    total_loss = generator_loss + content_loss # + diversity_loss

    # next level features loss
    if next_level_created_features is not None:
        next_level_created_features_loss = criterion_next_level_features(features[features_to_train - 1], next_level_created_features)
        total_loss += next_level_created_features_loss

    generator.zero_grad()
    total_loss.backward()
    generator_optimizer.step()

    losses_dictionary = {'g_loss': generator_loss, 'content_loss': content_loss, 'total_loss': total_loss}
    # losses_dictionary['diversity loss'] = diversity_loss
    if next_level_created_features is not None:
        losses_dictionary['next_level_loss'] = next_level_created_features_loss
    return losses_dictionary


def train_discriminator(generator, discriminator, discriminator_loss_fn, discriminator_optimizer, real_images, features, features_to_train):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(real_images.shape[0], 128, 1, 1).to(device)
    fake_images = generator(z, features, features_to_train)[0].detach()

    real_images_d_logit = discriminator(real_images)
    fake_images_d_logit = discriminator(fake_images)

    real_images_d_loss, fake_images_d_loss = discriminator_loss_fn(real_images_d_logit, fake_images_d_logit)
    gp = gradient_penalty(functools.partial(discriminator), real_images, fake_images)

    discriminator_loss = (real_images_d_loss + fake_images_d_loss) + gp * args.gradient_penalty_weight

    discriminator.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()

    return {'d_loss': real_images_d_loss + fake_images_d_loss, 'gp': gp}


def sample(generator, z, features, features_level):
    with torch.no_grad():
        generator.eval()
        return generator(z, features, features_level)[0]


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
    normalizer_discriminator = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    classifier = torch.load("../try/classifier18")
    classifier.eval()
    generator = Generator(gen_type=args.gen_type)
    discriminator = Discriminator(args.discriminator_norm, dis_type=args.gen_type)
    # generator.load_state_dict(torch.load('./expc/expcG'))
    # discriminator.load_state_dict(torch.load('./expc/expcD'))
    classifier.to(device)
    generator.to(device)
    discriminator.to(device)

    # weights init
    generator.init_weights()
    discriminator.init_weights()

    # losses + optimizers
    criterion_discriminator, criterion_generator = get_wgan_losses_fn()
    criterion_features = nn.L1Loss()
    criterion_diversity_n = nn.L1Loss()
    criterion_diversity_d = nn.L1Loss()
    criterion_next_level_features = nn.L1Loss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    num_of_epochs = args.epochs

    starting_time = time.time()
    iterations = 0
    outputs_dir = os.path.join('wgan-gp_models', args.model_name)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
    temp_results_dir = os.path.join(outputs_dir, 'temp_results')
    if not os.path.isdir(temp_results_dir):
        os.mkdir(temp_results_dir)
    models_dir = os.path.join(outputs_dir, 'models_checkpoint')
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    writer = tensorboardX.SummaryWriter(os.path.join(outputs_dir, 'summaries'))
    z = torch.randn(args.batch_size, 128, 1, 1).to(device)  # a fixed noise for sampling
    # z2 = torch.randn(args.batch_size, 128, 1, 1).to(device)  # a fixed noise for diversity sampling
    fixed_features = 0
    # fixed_features_diversity = 0
    first_iter = True
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        for data in train_loader:
            iterations += 1
            if iterations % 30 == 1:
                print('epoch:', epoch, ', iter', iterations, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.to(device)  # change to gpu tensor
            images_discriminator = normalizer_discriminator(images)
            images_clf = normalizer_clf(images)
            _, features = classifier(images_clf)
            features = list(features)
            if first_iter:
                first_iter = False
                fixed_features = [torch.clone(features[x]) for x in range(len(features))]
                # fixed_features_diversity = [torch.clone(features[x]) for x in range(len(features))]
                # for i in range(len(features)):
                #     for j in range(fixed_features_diversity[i].shape[0]):
                #         fixed_features_diversity[i][j] = fixed_features_diversity[i][j % 8]
                grid = vutils.make_grid(images_discriminator, padding=2, normalize=True, nrow=8)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'original_images.jpg'))
                # orig_images_diversity = torch.clone(images_discriminator)
                # for i in range(orig_images_diversity.shape[0]):
                #     orig_images_diversity[i] = orig_images_diversity[i % 8]
                # grid = vutils.make_grid(orig_images_diversity, padding=2, normalize=True, nrow=8)
                # vutils.save_image(grid, os.path.join(temp_results_dir, 'original_images_diversity.jpg'))
            features_to_train = 1
            if iterations < 3000:
                features_to_train = 1
            elif iterations < 8000:
                features_to_train = random.randint(1, 2)
            elif iterations < 16000:
                features_to_train = random.randint(1, 3)
            else:
                features_to_train = random.randint(1, 4)

            # for i in range(len(features)):
            #     if i < features_to_train:
            #         features[i] = features[i] * 0
            discriminator_loss_dict = train_discriminator(generator, discriminator, criterion_discriminator, discriminator_optimizer, images_discriminator, features, features_to_train)
            for k, v in discriminator_loss_dict.items():
                writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=iterations)
                if iterations % 30 == 1:
                    print('{}: {:.6f}'.format(k, v))
            if iterations % args.discriminator_steps == 1:
                generator_loss_dict = train_generator(generator, discriminator, criterion_generator, generator_optimizer, images.shape[0], features,
                                                      criterion_features, features_to_train, classifier, normalizer_clf, criterion_diversity_n,
                                                      criterion_diversity_d, criterion_next_level_features)
                for k, v in generator_loss_dict.items():
                    writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=iterations//5 + 1)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))

            if iterations < 10000 and iterations % 1000 == 1 or iterations % 2000 == 1:
                torch.save(generator.state_dict(),  models_dir + '/' + args.model_name + 'G')
                torch.save(discriminator.state_dict(), models_dir + '/' + args.model_name + 'D')
                # regular sampling (64 different images)
                first_features = True
                fake_images = None
                # fake_images_diversity = None
                for features_level in range(1, 5):
                    one_level_features = list(fixed_features)
                    # one_level_features_diversity = list(fixed_features_diversity)
                    # zero all features excepts the i'th level features
                    # for j in range(1, 5):
                    #     if j < features_level:
                    #         one_level_features[j] = one_level_features[j] * 0
                            # one_level_features_diversity[j] = one_level_features_diversity[j] * 0
                    if first_features:
                        first_features = False
                        fake_images = sample(generator, z, one_level_features, features_level)
                        # fake_images_diversity = sample(generator, z, one_level_features_diversity)
                    else:
                        tmp_fake_images = sample(generator, z, one_level_features, features_level)
                        fake_images = torch.vstack((fake_images, tmp_fake_images))
                        # tmp_fake_images = sample(generator, z2, one_level_features_diversity)
                        # fake_images_diversity = torch.vstack((fake_images_diversity, tmp_fake_images))
                grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=8)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'res_iter_{}.jpg'.format(iterations // 1000)))
                # diversity sampling (8 different images each with 8 different noises)
                # grid = vutils.make_grid(fake_images_diversity, padding=2, normalize=True, nrow=8)
                # vutils.save_image(grid, os.path.join(temp_results_dir, 'div_iter_{}.jpg'.format(iterations // 1000)))

            if iterations % 20000 == 1:
                torch.save(generator.state_dict(), models_dir + '/' + args.model_name + 'G_' + str(iterations // 15000))
                torch.save(discriminator.state_dict(), models_dir + '/' + args.model_name + 'D_' + str(iterations // 15000))


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
    out_dir = './wgan-gp_models/' + args.model_name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    arg_dic = vars(args)
    with open(out_dir + '/hyperparams.txt', 'w') as file:
        for key in arg_dic:
            file.write(key + ': {}\n'.format(arg_dic[key]))

    _main()
