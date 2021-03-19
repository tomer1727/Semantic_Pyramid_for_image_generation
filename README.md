# Semantic Pyramid for Image Generation
Implementation of semantic pyramid for image generation paper: https://semantic-pyramid.github.io/paper.pdf

# Introduction:
We present a GAN based model that takes advantage of the ability of deep convolution neural networks to extract features and attributes from images in order to classify images to classes, something proved to be successful. The idea is to use these features for the purpose of generating various new synthetic, real looking images. In addition, we would like to take advantage of the CNN structure so we will have control on the proximity of the generated images to the original ones. 
The implementation process is: 
An image is fed to a pre-trained classification network. It's features from different layers are extracted, and optionally manipulated according to the wanted application. The manipulated features are inserted to a GAN. Then, the generator creates synthetic images based on the manipulated features it has received. The user gets the ability to control the semantic level of features that is inserted to the generator.

# Train_model script:
 The main script, is used for model training. The following command line arguments are necessary in order to run:
 - model-name: the name of the current model (can be anything, for personal use).
 - classifier-path: the path to the classifier weights (the classifier network is included in this repository - classifier18 file).
 - train-path: the path to the train dataset.

 In addition there are several optional arguments. {choises} [deafult values]:
 - batch-size [16]
 - lr: learning rate [0.0002]
 - epochs [100]
 - discriminator-norm {batch_norm,instance_norm,layer_norm} [instance_norm]
 - gradient-penalty-weight [10.0]
 - discriminator-steps [5]
 - gen-type: generator block type, default for one convolutinal layer in each block, res for residual block {default, res} [res]
 - generator-path: if path is given the generator initialize to the given model in this path
 - discriminator-path: if path is given the discriminator initialize to the given model in this path
 - train1-prob: probability for choosing type 1 training procedure (choose 1 in order to use only type 1 training procedure) [0.6]
 - fixed-layer: used to train the model using only this layer
 - keep-temp-results: set this active for keeping temporary output images from the training procedure.
 - use-diversity-loss: set this active in order to use diversity loss as well

For example: 
``` 
python3 train_model.py  [-h]  [--batch-size BATCH_SIZE]
                              [--lr LR]
                              [--epochs EPOCHS] 
                              [--model-name MODEL_NAME]
                              [--discriminator-norm {batch_norm,instance_norm,layer_norm}]
                              [--gradient-penalty-weight GRADIENT_PENALTY_WEIGHT]
                              [--discriminator-steps DISCRIMINATOR_STEPS]
                              [--gen-type {default,res}]
                              [--train-path TRAIN_PATH]
                              [--classifier-path CLASSIFIER_PATH]
                              [--generator-path GENERATOR_PATH]
                              [--discriminator-path DISCRIMINATOR_PATH]
                              [--train1-prob TRAIN1_PROB]
                              [--fixed-layer FIXED_LAYER ]
                              [--keep-temp-results] 
                              [--use-diversity-loss]

 ```
 
 # eval_model script
 
 The main testing script. with this script images can be created using pre-trained model. the script repeat the following process: save one batch of images, and create images based on one layer of features at a time. For each layer use several noises (argument) and save grid of output images The number of samples like this is an argument for the script.
necessary running arguments: 
- full-model-name: the full path to the pre-tained model
- eval-path: the path to the dataset where the images are sampled from
- classifier-path: the path to the classifier
- output-path: where to save the outputs
Optional arguments {choises} [deafult values]:
- batch-size [8]
- num-of-samples: how many batches to use [5]
- num-of-noises: how many noise vectors to use per image [4] 
- gen-type: the type of the pre-trained generator {default, res} [res]

for example:
```
python3 eval_model.py [-h] --full-model-name FULL_MODEL_NAME
                           --eval-path EVAL_PATH 
                           [--batch-size BATCH_SIZE]
                           [--num-of-samples NUM_OF_SAMPLES]
                           [--num-of-noises NUM_OF_NOISES] 
                           --output-path OUTPUT_PATH 
                           [--gen-type {default,res}]
                           [--classifier-path CLASSIFIER_PATH]
```

An example for one output grid of the script (for layer 3, batch size of 3 and 3 noises):

![image](https://user-images.githubusercontent.com/62801710/111813005-58934e00-88e1-11eb-9ce3-49eb5725be10.png)
with the followin origin output:

![image](https://user-images.githubusercontent.com/62801710/111813103-71036880-88e1-11eb-9795-1eb40b5bbd61.png)

 
 # Examples:
 Some examples of images created by our model:
- Images that has been created by passing only layer 3 features:

![image](https://user-images.githubusercontent.com/62801710/111793436-f11fd300-88cd-11eb-88d5-ca5beedb1020.png)
- Examples for colorizing black and white images by passing features from layer 1 and 2:
![image](https://user-images.githubusercontent.com/62801710/111793694-2debca00-88ce-11eb-828f-8ad4945ecf84.png)
- Examples for our masking system, keeping more details about small part of the image while modifing the rest:
![image](https://user-images.githubusercontent.com/62801710/111793859-58d61e00-88ce-11eb-8aec-1b589e9b2c7a.png)


