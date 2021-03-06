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
import math

from classifier import Classifier
from generator import Generator
from discriminator import Discriminator


def print_gpu_details():
    """
    print gpu details to make sure gpu available
    """
    print('Is GPU available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Num of GPUs:', torch.cuda.device_count())


################################################################
# WGAN-GP Loss functions and helper methods
################################################################

def get_wgan_losses_fn():
    """
    get the wgan-gp loss functions (both for generator and discriminator)
    """
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

################################################################
# Training and sampling methods
################################################################


def train_generator(generator, discriminator, generator_loss_fn, generator_optimizer, batch_size, features, criterion_features, features_to_train,
                    classifier, normalizer_clf, criterion_diversity_n, criterion_diversity_d, masks, train_type, epsilon=10e-4):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(batch_size, 128, 1, 1, device=device)

    fake_images = generator(z, features, masks)
    # wgan-gp loss
    fake_images_d_logit = discriminator(fake_images)
    generator_loss = generator_loss_fn(fake_images_d_logit)

    # features loss (reconstruction loss)
    fake_images_clf = normalizer_clf(fake_images)
    _, fake_features = classifier(fake_images_clf)
    features_loss = getLossByTrainType(features, masks, train_type, features_to_train, fake_features, criterion_features)

    total_loss = generator_loss + features_loss
    loses_dictionary = {'g_loss': generator_loss, 'features_loss': features_loss, 'total_loss': total_loss}

    # diversity loss
    if args.use_diversity_loss:
        z2 = torch.randn(batch_size, 128, 1, 1, device=device)
        fake_images2 = generator(z2, features, masks)
        diversity_loss = criterion_diversity_n(z, z2) / (criterion_diversity_d(fake_images, fake_images2) + epsilon)
        total_loss = total_loss + diversity_loss
        loses_dictionary['diversity loss'] = diversity_loss
        loses_dictionary['total loss'] = total_loss

    # total loss calculation and network updating
    generator.zero_grad()
    total_loss.backward()
    generator_optimizer.step()

    return loses_dictionary


def train_discriminator(generator, discriminator, discriminator_loss_fn, discriminator_optimizer, real_images, features, masks):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(real_images.shape[0], 128, 1, 1).to(device)
    fake_images = generator(z, features, masks).detach()

    # wgan-gp loss
    real_images_d_logit = discriminator(real_images)
    fake_images_d_logit = discriminator(fake_images)

    real_images_d_loss, fake_images_d_loss = discriminator_loss_fn(real_images_d_logit, fake_images_d_logit)
    gp = gradient_penalty(functools.partial(discriminator), real_images, fake_images)

    discriminator_loss = (real_images_d_loss + fake_images_d_loss) + gp * args.gradient_penalty_weight

    # network updating
    discriminator.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()

    return {'d_loss': real_images_d_loss + fake_images_d_loss, 'gp': gp}


def sample(generator, z, features, masks):
    """
    generate a batch of images from the given (Z, F, M)
    """
    with torch.no_grad():
        generator.eval()
        return generator(z, features, masks)


def isolate_layer(fixed_features, selected_layer, device):
    """
    keep only the selected layer and zero the rest layers
    """
    res = [torch.clone(fixed_features[x]) for x in range(len(fixed_features))]
    for i in range(len(fixed_features)):
        if i != selected_layer:
            res[i] = torch.zeros(res[i].shape, device=device)
    return res

################################################################
# Masks handling methods
################################################################


def setCroppedMask(random_crop, mask, window_len1=0.45, window_len2=0.45, window_data=0):
    """
    Calculate window size and mask init accordingly
    """
    x_start = math.floor((random_crop[0]/100) * mask.shape[2])
    y_start = math.floor((random_crop[1]/100) * mask.shape[3])
    x_end = x_start + math.floor(window_len1 * mask.shape[2])
    y_end = y_start + math.floor(window_len2 * mask.shape[3])
    mask[:, :, x_start:x_end, y_start:y_end] = window_data
    return mask


def setMasksPart1(masks, device, random_layer_idx):
    """
    Set masks suits to train type 1
    """
    for idx, mask in enumerate(masks):
        if idx == random_layer_idx:
            # Pass the whole feature layer
            masks[idx] = torch.ones(mask.shape, device=device)
        else:
            # Block the feature layer
            masks[idx] = mask*0


def setMasksPart2(masks, device, random_layer_idx):
    """
    Set masks suits to train type 1
    """
    random_crop = (random.randint(20, 40), random.randint(20, 40)) # randomize the percentage of mask indexes
    masks[random_layer_idx] = torch.ones(masks[random_layer_idx].shape, device=device)
    for idx, mask in enumerate(masks):
        if idx < random_layer_idx:
            # Pass a part of the feature layer
            masks[idx] = setCroppedMask(random_crop, torch.ones(mask.shape, device=device))
        else:
            # Block the feature layer
            masks[idx] = mask*0


def getLossByTrainType(features, masks, train_type, features_to_train, outputs_images_features, criterion_features):
    """
    Calculate the features loss according to the train type
    """
    if train_type == 1:
        loss_features = criterion_features(features[features_to_train], outputs_images_features[features_to_train])
    else:
        loss_features = criterion_features(features[1]*masks[1], outputs_images_features[1]*masks[1])
        for i in range(2, features_to_train + 1):
            loss_features += criterion_features(features[i]*masks[i], outputs_images_features[i]*masks[i])  # calculate loss
    return loss_features*(1/features_to_train)*200


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

    # Network creation
    classifier = torch.load(args.classifier_path)
    classifier.eval()
    generator = Generator(gen_type=args.gen_type)
    discriminator = Discriminator(args.discriminator_norm, dis_type=args.gen_type)
    # init weights
    if args.generator_path is not None:
        generator.load_state_dict(torch.load(args.generator_path))
    else:
        generator.init_weights()
    if args.discriminator_path is not None:
        discriminator.load_state_dict(torch.load(args.discriminator_path))
    else:
        discriminator.init_weights()

    classifier.to(device)
    generator.to(device)
    discriminator.to(device)

    # losses + optimizers
    criterion_discriminator, criterion_generator = get_wgan_losses_fn()
    criterion_features = nn.L1Loss()
    criterion_diversity_n = nn.L1Loss()
    criterion_diversity_d = nn.L1Loss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    num_of_epochs = args.epochs

    starting_time = time.time()
    iterations = 0
    # creating dirs for keeping models checkpoint, temp created images, and loss summary
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
    z2 = torch.randn(args.batch_size, 128, 1, 1).to(device)  # a fixed noise for diversity sampling
    fixed_features = 0
    fixed_masks = 0
    fixed_features_diversity = 0
    first_iter = True
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        for data in train_loader:
            train_type = random.choices([1, 2], [args.train1_prob, 1-args.train1_prob]) # choose train type
            iterations += 1
            if iterations % 30 == 1:
                print('epoch:', epoch, ', iter', iterations, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.to(device)  # change to gpu tensor
            images_discriminator = normalizer_discriminator(images)
            images_clf = normalizer_clf(images)
            _, features = classifier(images_clf)
            if first_iter: # save batch of images to keep track of the model process
                first_iter = False
                fixed_features = [torch.clone(features[x]) for x in range(len(features))]
                fixed_masks = [torch.ones(features[x].shape, device=device) for x in range(len(features))]
                fixed_features_diversity = [torch.clone(features[x]) for x in range(len(features))]
                for i in range(len(features)):
                    for j in range(fixed_features_diversity[i].shape[0]):
                        fixed_features_diversity[i][j] = fixed_features_diversity[i][j % 8]
                grid = vutils.make_grid(images_discriminator, padding=2, normalize=True, nrow=8)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'original_images.jpg'))
                orig_images_diversity = torch.clone(images_discriminator)
                for i in range(orig_images_diversity.shape[0]):
                    orig_images_diversity[i] = orig_images_diversity[i % 8]
                grid = vutils.make_grid(orig_images_diversity, padding=2, normalize=True, nrow=8)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'original_images_diversity.jpg'))
            # Select a features layer to train on
            features_to_train = random.randint(1, len(features) - 2) if args.fixed_layer is None else args.fixed_layer
            # Set masks
            masks = [features[i].clone() for i in range(len(features))]
            setMasksPart1(masks, device, features_to_train) if train_type == 1 else setMasksPart2(masks, device, features_to_train)
            discriminator_loss_dict = train_discriminator(generator, discriminator, criterion_discriminator, discriminator_optimizer, images_discriminator, features, masks)
            for k, v in discriminator_loss_dict.items():
                writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=iterations)
                if iterations % 30 == 1:
                    print('{}: {:.6f}'.format(k, v))
            if iterations % args.discriminator_steps == 1:
                generator_loss_dict = train_generator(generator, discriminator, criterion_generator, generator_optimizer, images.shape[0], features,
                                                      criterion_features, features_to_train, classifier, normalizer_clf, criterion_diversity_n,
                                                      criterion_diversity_d, masks, train_type)

                for k, v in generator_loss_dict.items():
                    writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=iterations//5 + 1)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))

            # Save generator and discriminator weights every 1000 iterations
            if iterations % 1000 == 1:
                torch.save(generator.state_dict(), models_dir + '/' + args.model_name + 'G')
                torch.save(discriminator.state_dict(), models_dir + '/' + args.model_name + 'D')
            # Save temp results
            if args.keep_temp_results:
                if iterations < 10000 and iterations % 1000 == 1 or iterations % 2000 == 1:
                    # regular sampling (batch of different images)
                    first_features = True
                    fake_images = None
                    fake_images_diversity = None
                    for i in range(1, 5):
                        one_layer_mask = isolate_layer(fixed_masks, i, device)
                        if first_features:
                            first_features = False
                            fake_images = sample(generator, z, fixed_features, one_layer_mask)
                            fake_images_diversity = sample(generator, z, fixed_features_diversity, one_layer_mask)
                        else:
                            tmp_fake_images = sample(generator, z, fixed_features, one_layer_mask)
                            fake_images = torch.vstack((fake_images, tmp_fake_images))
                            tmp_fake_images = sample(generator, z2, fixed_features_diversity, one_layer_mask)
                            fake_images_diversity = torch.vstack((fake_images_diversity, tmp_fake_images))
                    grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=8)
                    vutils.save_image(grid, os.path.join(temp_results_dir, 'res_iter_{}.jpg'.format(iterations // 1000)))
                    # diversity sampling (8 different images each with few different noises)
                    grid = vutils.make_grid(fake_images_diversity, padding=2, normalize=True, nrow=8)
                    vutils.save_image(grid, os.path.join(temp_results_dir, 'div_iter_{}.jpg'.format(iterations // 1000)))

                if iterations % 20000 == 1:
                    torch.save(generator.state_dict(), models_dir + '/' + args.model_name + 'G_' + str(iterations // 15000))
                    torch.save(discriminator.state_dict(), models_dir + '/' + args.model_name + 'D_' + str(iterations // 15000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model-name')
    parser.add_argument('--discriminator-norm', default='instance_norm', choices=['batch_norm', 'instance_norm', 'layer_norm'])
    parser.add_argument('--gradient-penalty-weight', type=float, default=10.0)
    parser.add_argument('--discriminator-steps', type=int, default=5)
    parser.add_argument('--gen-type', default='res', choices=['default', 'res'])
    parser.add_argument('--train-path', default='/home/dcor/datasets/places365')
    parser.add_argument('--classifier-path', default='/home/dcor/ronmokady/workshop21/team1/try/classifier18')
    parser.add_argument('--generator-path', help='None for random init, if path is given the generator initialize to the given model')
    parser.add_argument('--discriminator-path', help='None for random init, if path is given the discriminator initialize to the given model')
    parser.add_argument('--train1-prob', default=0.6, type=float, help='use value of 1 in order of use only train type 1 (no masks)')
    parser.add_argument('--fixed-layer', type=int, help='train only this layer')
    parser.add_argument('--keep-temp-results', action='store_true')
    parser.add_argument('--use-diversity-loss', action='store_true')
    args = parser.parse_args()
    print(args)
    if args.model_name is None:
        print("Must specify model name")
        exit(1)
    out_dir = './wgan-gp_models/' + args.model_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    arg_dic = vars(args)
    with open(out_dir + '/hyperparams.txt', 'w') as file:
        for key in arg_dic:
            file.write(key + ': {}\n'.format(arg_dic[key]))

    _main()
