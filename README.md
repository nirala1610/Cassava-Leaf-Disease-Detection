# Cassava Leaf Disease Detection
Cassava is the most important tropical root crop. Its starchy roots are 
a major source of dietary energy for more than 500 million people. It 
is known to be the highest producer of carbohydrates among staple 
crops. It is a very resistant crop however it is not immune to all viral 
and bacterial diseases. Cassava Mosaic Disease alone causes an 
annual loss of US$ 1.2 to 2.3 billion. It is a very concerning issue 
hence it is imperative to detect the disease at its early stages. Modern 
deep learning techniques can be very useful given the need for 
predicting diseases with high precision. We propose one-such deep 
learning-based solution involving Convolutional Neural Networks. 
Our work proposes a methodology for solving the task of classifying 
the four most prevalent types of cassava leaf diseases and healthy 
cassava leaves. For this we trained five different classifiers for the 
five different classes, each classifier used the model as the base model 
followed by few fully connected layers. The model using the original 
EfficientNetB0, DenseNet, VGG16, InceptionNetV3 architecture. In our 
observation DenseNet and EfficientNetB0 give better accuracy as 
compare to other model. Further, we 
deployed the EfficientNetB0 model on Android Application using Android 
Studio, Java and XML for more accessibility of our classifier.


## Dataset
Images of cassava leaves were obtained from the Kaggle competition. Cassava
consists of leaf images for the cassava plant depicting healthy and four (4) 
disease conditions; Cassava Mosaic Disease (CMD), Cassava Bacterial Blight 
(CBB), Cassava Green Mottle (CGM) and Cassava Brown Streak Disease 
(CBSD).
The dataset consists of 21,397 labelled images collected during a regular survey 
in Uganda. Most images were crowdsourced from farmers taking photos of their 
gardens, and annotated by experts at the National Crops Resources Research 
Institute (NCRRI) in collaboration with the AI lab at Makerere University, 
Kampala.

It includes 21,397 training images categorized into five distinct classes
A. Cassava Bacterial Blight (CBB) – 1087 images 
B. Cassava Brown Streak Disease (CBSD) – 2189 images
C. Cassava Green Mottle (CGM) – 2386 images 
D. Cassava Mosaic Disease (CMD) – 13158 images 
E. Healthy – 2577 images.
Picture of cassava leaves as a dataset for each class is shown in following 
figures : A, B, C, D and E.

<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/ec9d1806-a133-4d9d-8070-d1b535bdee1d" width= "700" Height = "700" >

## Data pre-processing
Due to avoid biasing in training we took equal number of images (1087) in each 
class. In conducting experiments, the dataset is divided into three parts, 70% 
training data, 20% validation data, and 10% test data of the total datasets used.
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/26b35916-d7d8-4514-be84-8f71ab2772d2" width= "30%" Height = "30%" >

##  Training Process:
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/c096daf8-32a3-43c6-b09e-b0d5d257fe67" width= "50%"   >

## Cassava leaf Disease Predictor Android application
 splash Interface
<p align="center">
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/57f9cf10-1e80-4ffe-a93b-299ea8d42f07" width= "30%" Height = "10%" >
 
</p>
Before Image upload
<p align="center">
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/3d6ad621-8139-4eb6-90f5-52347a094757"  width= "30%" Height = "10%" >
  </p>
  
  After Image upload	
 <p align="center">
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/b4c0249a-17d7-42bb-9d0a-f76ed4edc4b1"  width= "30%" Height = "10%" >
  </p>
  
  Result Predict
 <p align="center">
<img src="https://github.com/nirala1610/Leaf-Disease-Detection/assets/93898811/fa87cb3b-cfde-408c-9a18-ccdb0ed112f7"  width= "30%" Height = "10%" >
  </p>
