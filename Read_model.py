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

test = pd.read_csv('test.txt',header=None, sep=' ',encoding='gb2312')
test.columns = ['path']

img_matrix = imread(test.iloc[0, :].path)

height, width, depth = imresize(img_matrix, (224, 224)).shape

test_number = len(test)


for i in range(test_number):
    img_matrix = imread(test.iloc[i, :].path)
    img = imresize(img_matrix, (224, 224))
    x_test = (i,img)
    

model = load_model('my_model.h5')
preds = model.predict(x_test)

result = np.argmax(preds,axis = 1)
print(result.shape,result) 

#np.savetxt('1234.txt', result.reshape(1, result.shape[0]))
#result.write('test.txt'+'\n')
np.savetxt('result123.csv', result, fmt="%d",delimiter=",")