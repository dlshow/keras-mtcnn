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
import gc
import keras
import os
import msgpack
import msgpack_numpy as m
import tables

import sys
sys.path.append('/home/wk/e/mtcnn/keras-mtcnn/')
from MTCNNx import masked_cls, masked_bbox, masked_landmark, combine_cls_bbox_landmark
S=48
if os.path.exists('cache{}.h5'.format(S)):
    h5 = tables.open_file('cache48.h5', mode='r')
    ims_all = h5.root.ims.read()
    labels_all = h5.root.labels.read()
    h5.close()
    
else:
    with open(r'../48net/48/pts.imdb', 'rb') as fid:
        pts = pickle.load(fid)
    with open(r'../48net/48/cls.imdb','rb') as fid:
        cls = pickle.load(fid)
    with open(r'../48net/48/roi.imdb', 'rb') as fid:
        roi = pickle.load(fid)
    ims_cls = []
    ims_pts = []
    ims_roi = []
    cls_score = []
    pts_score = []
    roi_score = []
    for (idx, dataset) in enumerate(cls) :
        ims_cls.append( np.swapaxes(dataset[0],0,2))
        cls_score.append(dataset[1])
    for (idx,dataset) in enumerate(roi) :
        ims_roi.append( np.swapaxes(dataset[0],0,2))
        roi_score.append(dataset[2])
    for (idx,dataset) in enumerate(pts) :
        ims_pts.append( np.swapaxes(dataset[0],0,2))
        pts_score.append(dataset[3])

    ims_cls = np.array(ims_cls)
    ims_pts = np.array(ims_pts)
    ims_roi = np.array(ims_roi)
    cls_score = np.array(cls_score)
    pts_score = np.array(pts_score)
    roi_score = np.array(roi_score)
    one_hot_labels = to_categorical(cls_score, num_classes=2)
    gc.collect()

    ims_all,labels_all = combine_cls_bbox_landmark(ims_cls, one_hot_labels, ims_roi, roi_score, ims_pts, pts_score)
    del ims_cls, one_hot_labels, ims_roi, roi_score, ims_pts, pts_score

    h5 = tables.open_file('cache48.h5', mode='w', title='All')
    h5.create_array(h5.root, 'ims', ims_all)
    h5.create_array(h5.root, 'labels', labels_all)
    h5.close()


from MTCNNx import create_Kao_Onet
model = create_Kao_Onet(r'model48.h5')

lr = 0.01
batch_size = 1024*10
for i_train in range(10):
    print('round ', i_train)
    lr = lr * 0.5
    my_adam = adam(lr = lr)
    loss_list = {
        'cls':masked_cls,
        'bbox':masked_bbox,
        'landmark':masked_landmark
    }
    loss_weights_list = {
        'cls': 1.0,
        'bbox': 0.5,
        'landmark': 0.0
    }
    metrics_list = {
        'cls':'accuracy',
        'bbox': masked_bbox,
        'landmark':masked_landmark
    }

    #parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    #parallel_model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    #parallel_model.fit([ims_all], [labels_all, labels_all, labels_all], batch_size=batch_size, epochs=1)
    model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    model.fit([ims_all], [labels_all, labels_all, labels_all], batch_size=batch_size, epochs=2)
    model.save_weights('model48.h5')
    print('model saved')
