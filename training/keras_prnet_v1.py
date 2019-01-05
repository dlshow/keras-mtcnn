from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import _pickle as pickle
import random
from keras.activations import relu
from keras.losses import mean_squared_error
import tensorflow as tf
#tf.enable_eager_execution()
import gc
import keras
import os

import sys
S = 12
sys.path.append('/home/wk/e/mtcnn/keras-mtcnn/')
from MTCNNx import masked_cls, masked_bbox, combine_cls_bbox
if os.path.exists('cache{}.ims'.format(S)) and os.path.exists('cache{}.labels'.format(S)):
    with open('cache{}.ims'.format(S), 'rb') as f:
        ims_all = pickle.load(f)
    with open('cache{}.labels'.format(S), 'rb') as f:
        labels_all = pickle.load(f)
else:
    with open(r'../{}net/{}/cls.imdb'.format(S,S),'rb') as fid:
        cls = pickle.load(fid)
        print(len(cls))
    with open(r'../{}net/{}/roi.imdb'.format(S,S), 'rb') as fid:
        roi = pickle.load(fid)
        print(len(cls))

    ims_cls = []
    ims_roi = []
    cls_score = []
    roi_score = []
    for (idx, dataset) in enumerate(cls) :
        ims_cls.append( np.swapaxes(dataset[0],0,2))
        cls_score.append(dataset[1])
    for (idx,dataset) in enumerate(roi) :
        ims_roi.append( np.swapaxes(dataset[0],0,2))
        roi_score.append(dataset[2])

    ims_cls = np.array(ims_cls)
    ims_roi = np.array(ims_roi)
    cls_score = np.array(cls_score)
    roi_score = np.array(roi_score)
    one_hot_labels = to_categorical(cls_score, num_classes=2)

    ims_all,labels_all = combine_cls_bbox(ims_cls, one_hot_labels, ims_roi, roi_score)
    with open('cache{}.ims'.format(S), 'wb') as f:
        pickle.dump(ims_all, f, protocol=4)
    with open('cache{}.labels'.format(S), 'wb') as f:
        pickle.dump(labels_all, f, protocol=4)

from MTCNNx import create_Kao_Pnet,create_Kao_Rnet
if S == 12:
    model = create_Kao_Pnet(r'model{}.h5'.format(S))
    batch_size = 1024*100
else:
    model = create_Kao_Rnet(r'model{}.h5'.format(S))
    batch_size = 1024*50
import keras.callbacks as cbks
class KerasDebug(cbks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('masked_cls'):
                print(logs[k])
lr = 0.001

for i_train in range(2):
    print('round ', i_train)
    lr = lr * 0.5
    my_adam = adam(lr = lr)
    loss_list = {
        'cls':masked_cls,
        'bbox':masked_bbox
    }
    loss_weights_list = {
        'cls': 1.0,
        'bbox': 0.5
    }
    metrics_list = {
        'cls':'accuracy',
        'bbox': masked_bbox
    }

    parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    parallel_model.fit([ims_all], [labels_all, labels_all], batch_size=batch_size, epochs=32)
    '''
    model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    model.fit([ims_all], [labels_all, labels_all], batch_size=batch_size, epochs=1)#, callbacks=[KerasDebug()])
    '''
    model.save_weights('model{}.h5'.format(S))
    print('model saved')
