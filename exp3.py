"""
Experiment in Section 3.4.3

H-score for Image classification

Reuire: File './val.txt' from [http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz]
      Folder '../ILSVRC2012_img_val' from [http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar]
      packages: pillow

The details of the models can be found at
https://keras.io/applications/
"""

from keras.applications.xception import Xception, preprocess_input, decode_predictions
# Here we use Xception as an example, other networks could be validated using the same way
# Reference: [https://keras.io/applications/]

from keras.models import Model
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import pandas as pd
import sys
import os
import time
from func import h_score

model = Xception()

num_imgs = 50000 # Number of images in the Validation dataset
img_size = 299  #224 for resnet & VGG16 & VGG19 & MobileNet & DenseNet, 299 for Xception & InceptionV3 & InceptionResNetV2
x_list = np.zeros((num_imgs, img_size, img_size, 3))

# """ Path of validation dataset of ILSVRC2012  """
val_img_path = '../ILSVRC2012_img_val/'
y_label = pd.read_csv('val.txt', sep = ' ', header = None)[1].as_matrix()
# val.txt is the label of validation dataset in ILSVRC2012, could be downloaded in http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

img_pathlist = np.sort(os.listdir(val_img_path))
img_index = range(num_imgs)
id = 0
for id_img in img_index:
    cur_img = img_pathlist[id_img]
    img_path = val_img_path + cur_img
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x_i = image.img_to_array(img)
    x_list[id] = x_i
    if id_img % 100 == 0:
        print(str(id_img) + '/' + str(len(img_pathlist)))
    id = id + 1
print(str(num_imgs) + "Images Loaded:" + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
x_list = preprocess_input(x_list)
print(str(num_imgs) + "Images Preprocessed:" + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
s_model = Model(inputs=model.input, outputs=[model.layers[-1].input, model.layers[-1].output])
[s, preds] = s_model.predict(x_list)
print(str(num_imgs) + "Calculation of s(x) Finished:" + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
y_pred1 = np.argmax(preds, axis = 1)
y_pred5 = np.argpartition(preds, -5)[:, -5:]
acc1 = np.sum(y_pred1 == y_label)/preds.shape[0] 
acc5 = np.sum(np.any(y_pred5 == y_label.reshape(-1, 1), axis = 1)) / preds.shape[0]
num_train = 1.3e6
h = h_score(s, y_label)
trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
h_aic = h - trainable_count / num_train
print('acc1 = ', acc1)
print('acc5 = ', acc5)
print('h-score = ', h)
print('h-aic = ', h_aic)
