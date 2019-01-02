[//]: # (Image References)
[network]: ./images/fcn_network.png
[result]: ./images/training_result.png

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project Solution ##
Please follow this link for detailed steps for setting up the environment: [link](https://github.com/udacity/RoboND-DeepLearning-Project/blob/master/README.md)

### Network Architecture
For pixel-wise classification in this project, Fully Convolutional Network is used to detect the pixels that belong to our target “hero” object. Compared to CNN, FCN retain spatial information which is helpful in identifying the pixels as well as their location in the image. It consists of two main parts, encoder and decoder. Below is the diagram of the FCN network I have used in this project.

![alt text][network]

### Elements of FCN pipeline
#### 1. Encoder
This is a series of convolution layers which generate sets of features. It consists of four alternate convolution layers, each followed by a max pooling layer to down-sample the data (so each subsequent layer looks at a lower resolution image) and enhances dominant features by selecting the maximum value. This down-sampling also results in better generalization of the model so it is able to correctly classify the unforeseen images.

#### 2. 1x1 Convolution
The data from first convolution layers goes into the 1x1 convolution layer, which aggregates all the feature maps so far to generate a final set of features for classification. This 1x1 convolution acts just like a normal fully connected layer as it generates same number of features, but it differs in that it retains the spatial data regarding the generated features. Using this also has an additional benefit i.e. image of any size can be fed to this network.

#### 3. Decoder
The decoder part is responsible for performing up-sampling and reverse convolution on the generated features so to generate an output image of same size as that of input.
 
#### 4. Skip connections
Skip connections is a method in which a layer in encoder block is connected to a corresponding layer in the decoder block in a way to retain some features.  

#### 5. Fully connected layer
Finally the extrapolated feature set is connected to a fully connected layer in the end, which classifies the pixels into required classes.

### Training the FCN
To configure a network to detect target object, the network needs to be trained with the images of target object. For training the network, we need to feed labeled image data to the network. This labeled data is provided in the form of mask images in data/train/masks where the target object is labeled as blue pixels.

This means that the network is able to detect only objects it is trained with and to detect any other object (dog, cat, car etc), we need to retrain the network with the labeled images of the desired target.

For now, I am using the default images provided with the project but a better way is to generate the image data by running the Quad simulator.

### Configuring the network
#### 1. Number of layers
The selection of the number of network layers and filters is purely experimental. Generally, three convolution layers are sufficient to detect high-level features. For this project, I tried both three and four layers and didn’t see any significant improvement with four layers, which can be attributed to the limited test data. A test data set, with large number of images of target object in different scenarios (viewing angle, distance, crowd, other objects) might be able to make better use of the extra layer.     

#### 2. Filter size
I played around with different settings for filter depth for each layer. I found out that the training loss decreases with deeper convolution layers, but due to consequent higher number of parameters, the validation loss also increases, due to over-fitting.    

#### 3. Hyerparameters
The values of the below hyperparameters were found using a trial and error method.

* learning_rate = 0.001
* batch_size = 13
* num_epochs = 50
* steps_per_epoch = 336
* validation_steps = 40
* workers = 8

Below are some considerations kept in mind while tunining the hyperparameters.
##### Learning rate
Learning rate corresponds to the incremental correction in weights and bias. Using a very large learning rate results in faster convergence but usually higher loss. I experimented with both 0.01 and 0.001 and found 0.001 to yield lower loss.

##### Batch size
Batch size refers to the number of images that goes into the network in a single iteration and results in weights update. The value for batch size ranges from 1 to total number of images. The value equal to total images in a dataset results in pure gradient descent while value less than total dataset results in stochastic gradient descent. Using a large batch size for a very large dataset is very computationally intensive so its value is always a trade-off between accuracy and processing speed. For this project, I have selected the batch size to be 13.

##### Epoch
Epoch refers to a one iteration in which the complete data set has gone through the network. A good criteria for determining the epoch number is by observing the training loss in the network and stopping if there is no further significant improvement. I found experimentally that the value of loss stablizes after 50 epochs. 
  
##### Steps per epoch
Steps per epoch is roughly the total training images divided by batch size. The default test data consists of 4131 images. 

##### Validation steps
Validation steps is total validation images divided by batch size.

##### Workers
Workers refers to the number of parallel threads that perform computation.

### Results
I have managed to achieve a score of 0.433555988065 with current settings.

For detailed analysis, please visit the jupyter notebook [here](https://github.com/mykhani/RoboND-DeepLearning-Project/blob/master/code/model_training.ipynb).

Here is the result of training.
![alt text][result]
### Future Enhancements:
* It was observed that increasing the number of layers from 3 to 4 didn’t have a significant effect on the training/validation loss. This points out to the fact that the available data is limited to take full advantage of extra layer added. So, in future I would generate data from the quad simulator, consisting of different scenarios, instead of relying on the default provided data.
* I will try increasing the batchsize and see it's result.
* I will explore some advanced CNN architecture like inception module and see how it compares to my existing solution.
