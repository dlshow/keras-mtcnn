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
import tables

S = 48
import sys
sys.path.append('/home/wk/e/mtcnn/keras-mtcnn/')
from MTCNNx import create_Kao_Onet
from MTCNNx import masked_cls, masked_bbox, masked_landmark, combine_cls_bbox, combine_cls_bbox_landmark

def load_or_cache(cache_file, cache_file_mini, cls_file, roi_file, pts_file, load_all = True):
    #cache_file = 'cache{}w.h5'.format(S)
    if load_all == False and os.path.exists(cache_file_mini):
        h5 = tables.open_file(cache_file_mini, mode='r')
        ims_all = h5.root.ims.read()
        labels_all = h5.root.labels.read()
        h5.close()
        return ims_all, labels_all
        
    if os.path.exists(cache_file):
        h5 = tables.open_file(cache_file, mode='r')
        ims_all = h5.root.ims.read()
        labels_all = h5.root.labels.read()
        h5.close()
    else:
        with open(pts_file,
                  #r'../48netw/48/pts.imdb',
                  'rb') as fid:
            pts = pickle.load(fid)
        with open(cls_file,
                  #r'../48netw/48/cls.imdb',
                  'rb') as fid:
            cls = pickle.load(fid)
        with open(roi_file,
                  #r'../48netw/48/roi.imdb',
                  'rb') as fid:
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

        ims_all,labels_all = combine_cls_bbox_landmark(ims_cls, one_hot_labels, ims_roi, roi_score, ims_pts, pts_score)
        del ims_cls, one_hot_labels, ims_roi, roi_score, ims_pts, pts_score
        h5 = tables.open_file(cache_file, mode='w', title='All')
        h5.create_array(h5.root, 'ims', ims_all)
        h5.create_array(h5.root, 'labels', labels_all)
        h5.close()
    if not os.path.exists(cache_file_mini):
        indices = np.arange(len(ims_all))
        np.random.shuffle(indices)
        indices = indices[:int(len(indices)/10)]
        ims_all_mini = ims_all[indices]
        labels_all_mini = labels_all[indices]

        h5 = tables.open_file(cache_file_mini, mode='w', title='All')
        h5.create_array(h5.root, 'ims', ims_all_mini)
        h5.create_array(h5.root, 'labels', labels_all_mini)
        h5.close()
        if load_all == False:
            return ims_all_mini, labels_all_mini
    return ims_all, labels_all

ims_all, labels_all = load_or_cache('cache48w.h5',
                                    'cache48wmini.h5',
                                    '../48netw/48/cls.imdb',
                                    '../48netw/48/roi.imdb',
                                    '../48netw/48/pts.imdb',
                                    load_all = False)
print(len(ims_all), len(labels_all))


model = create_Kao_Onet(r'model{}.h5'.format(S))

lr = 0.001
batch_size = 64
for i_train in range(2):
    print('round ', i_train)
    lr = lr * 0.5
    my_adam = adam(lr = lr)
    loss_list = {
        'cls':masked_cls,
        'bbox':masked_bbox,
        'landmark':masked_landmark,
    }
    loss_weights_list = {
        'cls': 0.5,
        'bbox': 0.25,
        'landmark':0.5,
    }
    metrics_list = {
        #'cls':'accuracy',
        #'bbox': masked_bbox_fix,
        #'landmark':'accuracy'
    }

    #parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    #parallel_model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    #parallel_model.fit([ims_all], [labels_all, labels_all], batch_size=batch_size, epochs=32)

    model.compile(loss=loss_list, optimizer = my_adam, loss_weights=loss_weights_list, metrics=metrics_list)
    model.fit([ims_all], [labels_all, labels_all, labels_all], batch_size=batch_size, epochs=2)

    model.save_weights('model{}.h5'.format(S))
    print('model saved')
