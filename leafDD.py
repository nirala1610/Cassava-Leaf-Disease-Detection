import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from keras.optimizers import Adam
import os, cv2, json



# ignoring warnings
import warnings
warnings.simplefilter("ignore")
# For easy acces to files
WORK_DIR = "../input/cassava-leaf-disease-classification/"
os.listdir(WORK_DIR)
with open('../input/cassava-leaf-disease-classification/label_num_to_disease_map.json', 'r') as file:
    labels = json.load(file)
    
labels




data = pd.read_csv(WORK_DIR + "train.csv")

data.head()

data.dtypes

#change for the ImageDatagen and flow_from_dataframe
data.label = data.label.astype("str")
data.dtypes
data.dtypes
data.label.value_counts()


We have 21397 images for training and don't have an equal number of photos for each class.
I don't know how to deal with the unbalanced image dataset so I'll leave it to the next version.


Image Visualization
Let's first visualize the general data set.
Visualize by class later


IMG_SIZE = 300
plt.figure(figsize=(15,12))
data_sample = data.sample(9).reset_index(drop=True)


for i in range(8):
    plt.subplot(2,4,i+1)
    
    img = cv2.imread(WORK_DIR + "train_images/" + data_sample.image_id[i])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.axis("off")
    plt.imshow(img)
    plt.title(labels.get(data_sample.label[i]))
    
plt.tight_layout()
plt.show()


#Image Preprocessing, Data Augmentetion
#ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation.
#flow_from_dataframe: Takes the dataframe and the path to a directory + generates batches.
 #The generated batches contain augmented/normalized data.


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
                                    validation_split=0.2,
                                    #dtype=None,
) \
        .flow_from_dataframe(
                            data,
                            directory = WORK_DIR + "train_images",
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
                                validation_split = 0.2
) \
        .flow_from_dataframe(
                            data,
                            directory = WORK_DIR + "train_images",
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (IMG_SIZE, IMG_SIZE),
                            class_mode = "categorical",
                            batch_size = 32,
                            shuffle = True,
                            #seed = 34,
                            subset = "validation")

valid_generator.class_indices



#Model


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
model_check = ModelCheckpoint(
                            "./firstTry.h5",
                            monitor = "val_loss",
                            verbose = 1,
                            save_best_only = True,
                            save_weights_only = False,
                            mode = "min")


early_stop= EarlyStopping(
                                monitor = "val_loss",
                                min_delta=0.001,
                                patience=3,
                                verbose=1,
                                mode="min",
                                #baseline=None,
                                restore_best_weights=False)


reduce_lr = ReduceLROnPlateau(
                                monitor="val_loss",
                                factor=0.1,
                                patience=3,
                                verbose=1,
                                mode="min",
                                min_delta=0.0001,
                                #cooldown=0,
                                #min_lr=0
)


model.compile(optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])
history = model.fit_generator(train_generator,
                            epochs = 10,
                            validation_data = valid_generator,
                             callbacks = [model_check,early_stop
plt.figure(figsize=(15, 5))
plt.plot(history.history['accuracy'], 'b*-', label="train_acc")
plt.plot(history.history['val_accuracy'], 'r*-', label="val_acc")

plt.grid()
plt.title("train_acc vs val_acc")

plt.ylabel("Accuracy")
plt.xlabel("Epochs")

plt.legend()
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(history.history['accuracy'], 'b*-', label="train_acc")
plt.plot(history.history['val_accuracy'], 'r*-', label="val_acc")

plt.grid()

plt.title("train_acc vs val_acc")

plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.legend()

plt.show()

#Prediction


import os
import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt
# model = keras.models.load_model("./firstTry.h5")
preds = []
ss = pd.read_csv(WORK_DIR + "sample_submission.csv")
for image in ss.image_id:
    img = keras.preprocessing.image.load_img(WORK_DIR + "test_images/" + image)
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.preprocessing.image.smart_resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    preds.append(np.argmax(prediction))

my_submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})

my_submission.to_csv('submission.csv', index=False) 

my_submission