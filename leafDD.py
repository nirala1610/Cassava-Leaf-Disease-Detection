import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, DenseNet121,MobileNetV2, EfficientNetB0,EfficientNetB4, EfficientNetB7
from keras.optimizers import Adam
import os, cv2, json

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

# from google.colab import drive
# drive.mount('/content/drive')




import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'




! kaggle competitions download -c cassava-leaf-disease-classification

! unzip '/content/cassava-leaf-disease-classification.zip'

with open("/content/label_num_to_disease_map.json", 'r') as file:
    labels = json.load(file)
    
labels

data = pd.read_csv("train.csv")

# New Section



# We have 21397 images for training and don't have an equal number of photos for each class.
data.head()

data.dtypes

#change for the ImageDatagen and flow_from_dataframe
data.label = data.label.astype("str")

data.label.value_counts()

d0 = data[data['label']=="0"]
d0 = d0.iloc[:1087,:]

d1 = data[data['label']=="1"]
d1 = d1.iloc[:1087,:]

d2 = data[data['label']=="2"]
d2 = d2.iloc[:1087,:]

d3 = data[data['label']=="3"]
d3 = d3.iloc[:1087,:]

d4 = data[data['label']=="4"]
d4 = d4.iloc[:1087,:]

data = d0
data = data.append(d1)
data = data.append(d2)
data = data.append(d3)
data = data.append(d4)



data.shape

data = data.reset_index(drop=True)


Dtrain = data.iloc[0:5000,:]
Dtest = data.iloc[5000:5435,:]

data = Dtrain

data.shape

Dtest.shape





# Image Visualization
# Let's first visualize the general data set.
# Visualize by class later



IMG_SIZE = 224




plt.figure(figsize=(15,12))
data_sample = data.sample(9).reset_index(drop=True)

for i in range(8):
    plt.subplot(2,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()





**Cassava Bacterial Blight (CBB)**

labels.get("0")

plt.figure(figsize=(15,12))
data_sample = data[data.label=="0"].sample(4).reset_index(drop=True)
for i in range(4):
    plt.subplot(1,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()





**Cassava Brown Streak Disease (CBSD)**

labels.get("1")

plt.figure(figsize=(15,12))
data_sample = data[data.label=="1"].sample(4).reset_index(drop=True)
for i in range(4):
    plt.subplot(1,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()





**Cassava Green Mottle (CGM)**

labels.get("2")

plt.figure(figsize=(15,12))
data_sample = data[data.label=="2"].sample(4).reset_index(drop=True)
for i in range(4):
    plt.subplot(1,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()



**Cassava Mosaic Disease (CMD)**

labels.get("3")

plt.figure(figsize=(15,12))
data_sample = data[data.label=="3"].sample(4).reset_index(drop=True)
for i in range(4):
    plt.subplot(1,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()



**Healthy**

labels.get("4")

plt.figure(figsize=(15,12))
data_sample = data[data.label=="4"].sample(4).reset_index(drop=True)
for i in range(4):
    plt.subplot(1,4,i+1)
    
    img = cv2.imread("train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()

data.head()


**Image Preprocessing, Data Augmentetion**

train_generator = ImageDataGenerator(
                                    #featurewise_center=False,                                    
                                    #samplewise_center=False,
                                    #featurewise_std_normalization=False,
                                    #samplewise_std_normalization=False, 
                                    #zca_whitening=False,
                                    #zca_epsilon=1e-06,
                                    #rotation_range=90,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    #brightness_range=None,
                                    shear_range=25,
                                    zoom_range=0.3,
                                    #channel_shift_range=0.0,
                                    #fill_mode="nearest",
                                    #cval=0.0,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    #rescale=None,
                                    #preprocessing_function=None,
                                    #data_format=None,
                                    validation_split=0.1,
                                    #dtype=None,
) \
        .flow_from_dataframe(
                            data,
                            directory = "train_images",
                            x_col = "image_id",
                            y_col = "label",
                            #weight_col = None,
                            target_size = (IMG_SIZE, IMG_SIZE),
                            #color_mode = "rgb",
                             #classes = None,
                            class_mode = "categorical",
                            batch_size = 32,
                            shuffle = True,
                            #seed = 34,
                            #save_to_dir = None,
                            #save_prefix = "",
                            #save_format = "png",
                            subset = "training",
                            #interpolation = "nearest",
                            #validate_filenames = True
)

valid_generator = ImageDataGenerator(
                                    validation_split = 0.1
) \
        .flow_from_dataframe(
                            data,
                            directory = "train_images",
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (IMG_SIZE, IMG_SIZE),
                            class_mode = "categorical",
                            batch_size = 32,
                            shuffle = True,
                            #seed = 34,
                            subset = "validation")

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
Dtest,
directory= "train_images",
x_col="image_id",
y_col=None,
batch_size=32,
#seed=42,
shuffle=False,
class_mode=None,
target_size=(IMG_SIZE,IMG_SIZE)
)

valid_generator.class_indices

#train_generator.shape

#valid_generator.shape

**Model**



def modelEfficientNetB0():
    
    model = models.Sequential()
    model.add(EfficientNetB0(include_top = False, weights = "imagenet",
                            input_shape=(IMG_SIZE,IMG_SIZE, 3)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(5, activation = "softmax"))
    
    return model 


model = modelEfficientNetB0()


model.summary()


      
     


from tensorflow.keras import utils

utils.plot_model(model)

model.compile(optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])
                                          
                                          

history = model.fit_generator(train_generator,
                            epochs = 50,
                            validation_data = valid_generator)

history.shape

import os.path
if os.path.isfile('/content/drive/MyDrive/effB0_20.h5')  is False:
    model.save('/content/drive/MyDrive/effB0_20.h5')

#csv_logger = tf.keras.callbacks.

from tensorflow.keras.models import load_model
new_model = load_model('/content/drive/MyDrive/effModel.h5')

#evoluation
score = model.evaluate(test_generator,verbose=0) 

score





import os
import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt

preds = []
ss = Dtest
for image in ss.image_id:
    img = keras.preprocessing.image.load_img("train_images/" + image)
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.preprocessing.image.smart_resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    preds.append(np.argmax(prediction))



#from sklearn import metrics
#pred = model.predict(preds)

list(preds)

preds = [str(x) for x in list(preds)]

from sklearn.metrics import confusion_matrix

cf = confusion_matrix( Dtest['label'], preds)
print(cf)


import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as atlas

Dtest['label']
preds
conf_matrix = (confusion_matrix(Dtest['label'], preds))

# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix,annot=True, cmap='Blues')

# labels the title and x, y axis of plot
fx.set_title(' Confusion Matrix \n\n');
fx.set_xlabel('Predicted Values \n')
fx.set_ylabel('Actual Values \n');

# labels the boxes
fx.xaxis.set_ticklabels(['CBB','CBSD','CGM','CMD','Healthy'])
fx.yaxis.set_ticklabels(['CBB','CBSD','CGM','CMD','Healthy'])


atlas.show()


# printing classification report 
print(metrics.classification_report(Dtest['label'], preds))




# ploting graph Train accuracy vs validation accuracy

plt.figure(figsize=(15, 5))
plt.plot(history.history['accuracy'], 'b*-', label="train_acc")
plt.plot(history.history['val_accuracy'], 'r*-', label="val_acc")
plt.grid()
plt.title("train_acc vs val_acc")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()


# ploting graph train loss vs validation loss

plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], 'b*-', label="train_loss")
plt.plot(history.history['val_loss'], 'r*-', label="val_loss")
plt.grid()
plt.title("train_loss - val_loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

