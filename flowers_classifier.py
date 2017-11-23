import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.optimizers import Adagrad
from keras.layers import  Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Input
import pickle
from keras.models import load_model

train = pd.read_csv('train.txt', sep=' ', header=None)
train.columns = ['path', 'label']
val = pd.read_csv('val.txt', sep=' ', header=None)
val.columns = ['path', 'label']

train_number = len(train)
val_number = len(val)
class_number = len(np.unique(train.label))

# create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=False,)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='Adagrad')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

img_matrix = imread(train.iloc[0, :].path)
#height, width, depth = (224,224,3)

y_train = train.label.values.reshape((-1, 1))
y_val = val.label.values.reshape((-1, 1))

y_train = np_utils.to_categorical(y_train, class_number).reshape((-1, class_number))
y_val = np_utils.to_categorical(y_val, class_number).reshape((-1, class_number))


for i in range(train_number):
    img_matrix = imread(train.iloc[i, :].path)
    img = imresize(img_matrix, (224, 224))
    x_train = (i,img)

for i in range(val_number):
    img_matrix = imread(val.iloc[i, :].path)
    img = imresize(img_matrix, (224, 224))
    x_val = (i,img)
    
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, nb_epoch=20, validation_split=0.2)
#preds = model.predict(x_val)

model.save('my_model.h5') 
#pickle.dump(model, open(saveFiles, 'wb'))
