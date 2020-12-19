import torchvision.datasets as datasets

places_dataset = datasets.Places365('/home/dcor/datasets/places365', small=True, download=True)
