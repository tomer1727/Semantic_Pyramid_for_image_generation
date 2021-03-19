import torch
import os
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import torchvision.utils as vutils

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

    generator = Generator(gen_type=args.gen_type)
    generator.load_state_dict(torch.load(args.full_model_name))
    generator.eval()
    generator.to(device)

    classifier = torch.load(args.classifier_path)
    classifier.eval()
    classifier.to(device)

    eval_root = args.eval_path
    image_size = 256
    cropped_image_size = 256
    eval_set = dset.ImageFolder(root=eval_root,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(cropped_image_size),
                                    transforms.ToTensor()
                                ]))

    normalizer_clf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print('set data loader')
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    output_images_dir = os.path.join(args.output_path, args.full_model_name.split('/')[-1])
    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir, exist_ok=True)
    print("Starting Eval Loop...")
    sample_num = 1
    for data in eval_loader:
        images, _ = data
        images = images.to(device)  # change to gpu tensor
        images_clf = normalizer_clf(images)
        _, features = classifier(images_clf)
        masks = [torch.ones(features[i].shape, device=device) for i in range(len(features))]
        grid = vutils.make_grid(images, padding=2, normalize=False, nrow=8)
        vutils.save_image(grid, os.path.join(output_images_dir, 'orig_{}.jpg'.format(sample_num)))
        for features_level in range(1, 5):
            one_level_features = [torch.clone(features[i]) for i in range(len(features))]
            for i in range(len(features)):
                if i != features_level:
                    one_level_features[i] = one_level_features[i] * 0
            output_images = None
            for i in range(4):
                z = torch.randn(args.batch_size, 128, 1, 1, device=device)
                with torch.no_grad():
                    generator.eval()
                    if output_images is None:
                        output_images = generator(z, features, masks)
                    else:
                        tmp_images = generator(z, features, masks)
                        output_images = torch.vstack((output_images, tmp_images))
            grid = vutils.make_grid(output_images, padding=2, normalize=True, nrow=8)
            vutils.save_image(grid, os.path.join(output_images_dir, 'sample_{}_f{}.jpg'.format(sample_num, features_level)))
            sample_num += 1
            if sample_num == args.num_of_samples:
                print("done")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose hyperparameters')
    parser.add_argument('--full-model-name', required=True)
    parser.add_argument('--eval-path', required=True)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-of-samples', default=5, type=int)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--gen-type', default='res', choices=['default', 'res'])
    parser.add_argument('--classifier-path', default='/home/dcor/ronmokady/workshop21/team1/try/classifier18')
    args = parser.parse_args()
    _main()
