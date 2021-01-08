import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional
import torch.nn as nn
from PIL import Image
import os


class Classifier(nn.Module):
    """
    The full Classifier class, build from pre-trained resnet50 classifier model
    return intermediate outputs as well
    """
    def __init__(self, model):
        super(Classifier, self).__init__()
        # Save params
        # Encoder layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.flat = nn.Flatten()
        self.fc = model.fc

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)
        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        avgpool = self.avgpool(layer4)
        flat = self.flat(avgpool)
        outputs = self.fc(flat)
        return outputs, (relu, layer1, layer2, layer3, layer4, avgpool)


def _main():
    # the classifier architecture
    arch = 'resnet18'

    # pre-trained weights file path
    model_file_dir = r"C:\Users\tomer\OneDrive\Desktop\sadna\resnet"
    model_file = os.path.join(model_file_dir, arch+"_places365.pth.tar")

    # use the weights to initialize the classifier
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    clf = Classifier(model)
    torch.save(clf, r"C:\Users\tomer\OneDrive\Desktop\sadna\pyramid_project\classifier18")
    # torch.save(clf, "./classifier")
    model.eval()
    clf.eval()

    # load the image transformer
    adjustImage = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = r"C:\Users\tomer\OneDrive\Desktop\sadna\categories_places365.txt"
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    #######################################################
    # Optional code:                                      #
    #######################################################

    # load the test image
    # img_name = r"C:\Users\tomer\OneDrive\Desktop\12.jpg"
    img_name = r"C:\Users\tomer\OneDrive\Desktop\00004953.jpg"

    img = Image.open(img_name)
    input_img = Variable(adjustImage(img).unsqueeze(0))

    # forward pass model
    logit = model.forward(input_img)
    h_x = functional.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('original model')
    print('{} prediction on {}'.format(arch,img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # forward pass resnet
    logit, features = clf.forward(input_img)
    h_x = functional.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('clf model')
    print('{} prediction on {}'.format(arch,img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    for f in features:
        print(f.shape)


if __name__ == '__main__':
    _main()