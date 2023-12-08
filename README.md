# Building a semantic segmentation model

Goal: to segment ships on satellite images. 

Method: U-net, Keras. 

Metric: Dice score

Dataset has more then 120000 images without ships and less images with 1-10 ships. 

## Use balanced part of all dataset

500 images with no ships, 1500 images on each of 7 groups of ships quantity.

### The best parameters searched:
Deconvolution with UpSample2d  
Gaussian noise 0.1    
4 blocks by 2 conv2d layers with 8-128 neurons and max pooling  
activation relu  
optimizer Adam  

### Metric did not increase than 0.3. 

Let will use all dataset without empty images (without ships). 

## Use only ship-include images and all dataset

The current neural network must perform two tasks: classification (there are or are not ships in the photo) and segmentation, if any. You can break these tasks down: one neural network predicts whether there are ships in the photo. And the other segments. Presumably, a network trained only on pictures of ships will have greater accuracy.


### The best parameters we searched:

Deconvolution with UpSample2d  
Image scaling with average pooling  
Gaussian noise 0.1  
4 blocks by 2 conv2d layers with 8-128 neurons and max pooling  
activation relu  
optimizer Adam  

### Best score we found: 0.4

# To improve the quality of the model we can use:

Search and kick off an outliers (bag satellite images)  

Image pre-processing such as Equalization, Contrast stretching, Adaptive Equalization  

Use a different metric. The problem is that there are very few pixels that ships occupy. This results in a strong imbalance of the positive and negative classes in one image. Analysis of publications shows that instead of the dice coefficient, you can use, for example, focal loss.  

The current neural network must perform two tasks: classification (there are or are not ships in the photo) and segmentation, if any. You can break these tasks down: one neural network predicts whether there are ships in the photo. And the other segments. Presumably, a network trained only on pictures of ships will have greater accuracy.   

Analysis of publications shows that the use of pre-trained models (for example, Unet34) improves the metric.  