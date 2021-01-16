import torch
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import time
import tensorboardX
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


def train_generator(generator, discriminator, generator_loss_fn, generator_optimizer, batch_size):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(batch_size, 128, 1, 1, device=device)

    fake_images = generator(z)
    fake_images_d_logit = discriminator(fake_images)
    generator_loss = generator_loss_fn(fake_images_d_logit)

    generator.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()

    return {'g_loss': generator_loss}


def train_discriminator(generator, discriminator, discriminator_loss_fn, discriminator_optimizer, real_images):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator.train()
    discriminator.train()

    z = torch.randn(real_images.shape[0], 128, 1, 1).to(device)
    fake_images = generator(z).detach()

    real_images_d_logit = discriminator(real_images)
    fake_images_d_logit = discriminator(fake_images)

    real_images_d_loss, fake_images_d_loss = discriminator_loss_fn(real_images_d_logit, fake_images_d_logit)
    gp = gradient_penalty(functools.partial(discriminator), real_images, fake_images)

    discriminator_loss = (real_images_d_loss + fake_images_d_loss) + gp * args.gradient_penalty_weight

    discriminator.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()

    return {'d_loss': real_images_d_loss + fake_images_d_loss, 'gp': gp}


def sample(generator, z):
    with torch.no_grad():
        generator.eval()
        return generator(z)


def _main():
    print_gpu_details()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    train_root = args.train_path

    image_size = 128
    cropped_image_size = 128
    print("set image folder")
    train_set = dset.ImageFolder(root=train_root,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(cropped_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                 ]))
    print('set data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    generator = Generator()
    discriminator = Discriminator(args.discriminator_norm)
    # generator.load_state_dict(torch.load('./expc/expcG'))
    # discriminator.load_state_dict(torch.load('./expc/expcD'))
    generator.to(device)
    discriminator.to(device)

    # weights init
    generator.init_weights()
    discriminator.init_weights()

    # losses + optimizers
    criterion_discriminator, criterion_generator = get_wgan_losses_fn()
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
    z = torch.randn(100, 128, 1, 1).to(device)  # a fixed noise for sampling
    print("Starting Training Loop...")
    for epoch in range(num_of_epochs):
        for data in train_loader:
            iterations += 1
            if iterations % 30 == 1:
                print('epoch:', epoch, ', iter', iterations, 'start, time =', time.time() - starting_time, 'seconds')
                starting_time = time.time()
            images, _ = data
            images = images.to(device)  # change to gpu tensor

            discriminator_loss_dict = train_discriminator(generator, discriminator, criterion_discriminator, discriminator_optimizer, images)
            for k, v in discriminator_loss_dict.items():
                writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=iterations)
                if iterations % 30 == 1:
                    print('{}: {:.6f}'.format(k, v))
            if iterations % args.discriminator_steps == 1:
                generator_loss_dict = train_generator(generator, discriminator, criterion_generator, generator_optimizer, images.shape[0])
                for k, v in generator_loss_dict.items():
                    writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=iterations//5 + 1)
                    if iterations % 30 == 1:
                        print('{}: {:.6f}'.format(k, v))

            # sample images from the fixed noise
            if iterations % 500 == 1:
                fake_images = sample(generator, z)
                grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=10)
                vutils.save_image(grid, os.path.join(temp_results_dir, 'res_iter_{}.jpg'.format(iterations // 500)))

            if iterations % 1000 == 1:
                torch.save(generator.state_dict(),  models_dir + '/' + args.model_name + 'G')
                torch.save(discriminator.state_dict(), models_dir + '/' + args.model_name + 'D')

            if iterations % 15000 == 1:
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
    parser.add_argument('--train-path', default='/home/dcor/datasets/places365')
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
