# Semantic-Segmentation
Binary and Multi Class Semantic Segmentation from scratch and using Transfer Learning.

The main goal of this project is to gain a fundamental understanding of how deep learning can be used in computer vision applications.

The code is provided in the Jupyter notebook Binary and Multi class Semantic Segmentation.ipynb

A segmentation task differs from classification in that we want to label every single pixel in
the image (not just one label for entire image). Imagine that we are classifying the central
pixel of the image (a 1x1 output) based on its surroundings (a NxN input). In order to label
every pixel in the image, we can reformulate one NxN image as being N2 pixel-wise examples,
each having a different pixel in the center that gets classified based on a surrounding window. Later, more parallel implementations closely follow the encoder-decoder structure,
by using convolutional layers to encode and transposed convolution (a.k.a. deconvolution)
layers to decode directly into pixel-wise predictions. The loss function which we
should have used for the classification task is then applied to and averaged across all the
pixels. From this, one of the most popular networks for image segmentation was derived by
adding so-called skip connections from encoding layers to their corresponding decoding layers
at the same scale to enhance learning in earlier layers and improve the reconstruction
of feature maps at a higher resolution from feature maps at a lower resolution.
In this project, we implement a segmentation CNN that extracts the objects from an image. 

## Create two segmentation models: 

### model 1 trained from scratch, 
### model 2 using transfer learning.

Unet and pre-trained VGG models are used in this project.
The dataset used is PASCAL VOC 2009 Dataset. This dataset consists of colour images of various scenes with different object classes (e.g. animal: bird, cat, ...; vehicle: aeroplane, bicycle, ...), totalling 20 classes (Figure 1). The dataset has been used as a benchmark for classification, detection and segmentation challenges and for comparing results of individual research.
