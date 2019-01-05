from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense,concatenate
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
import keras.backend as K
from keras.losses import mean_squared_error
import numpy as np
import os
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

def create_Kao_Onet( weight_path = 'model48.h5', train=True):
    '''input = Input(shape = [48,48,3])
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)
    '''
    #------------------------------------------------------------------
    input = Input(shape = [48,48,3])
    input_randx = Input(shape=[1])
    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)
    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)
    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(128,(3,3),strides=1,padding='valid',name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu',name='dense1') (x)
    classifier = Dense(2, activation='softmax',name='cls')(x)
    bbox_regress = Dense(4,name='bbox')(x)
    landmark_regress = Dense(10,name='landmark')(x)
    
    #model = Model([input], [classifier, bbox_regress, landmark_regress])
    #model.load_weights(weight_path, by_name=True)
    if train:
        crl = Model([input], [classifier, bbox_regress, landmark_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            crl.load_weights(weight_path, by_name=True)
        return crl
    else:
        crl = Model([input], [classifier, bbox_regress, landmark_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            crl.load_weights(weight_path, by_name=True)
        return crl

def create_Kao_Rnet (weight_path = 'model24.h5', train=True):
    '''
    input = Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    '''
    #------------------------------------------------------
    input = Input(shape = [24,24,3]) # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(16,(3,3),strides=1,padding='same',name='conv1')(input)
    c1out = PReLU(shared_axes=[1,2],name='prelu1')(x)
    c1out = concatenate ([c1out,input],axis=3)

    c2input = MaxPool2D(pool_size=3)(c1out)

    x = Conv2D(32,(3,3),strides=1,padding='same',name='conv2')(c2input)
    c2out = PReLU(shared_axes=[1,2],name='prelu2')(x)
    c2out = concatenate([c2out,c2input],axis=3)

    c3input = MaxPool2D(pool_size=2)(c2out)

    x = Conv2D(64,(3,3),strides=1,padding='same',name='conv3')(c3input)
    c3out = PReLU(shared_axes=[1,2],name='prelu3')(x)
    c3out = concatenate([c3out,c3input],axis=3)

    x = Flatten() (c3out)
    x = Dense(128,name='dense1')(x)
    x = PReLU(shared_axes=[1],name='prelu4')(x)
    classifier = Dense(2, activation='softmax',name='cls')(x)
    bbox_regress = Dense(4,name='bbox')(x)

    #model = Model([input], [classifier, bbox_regress])
    #model.load_weights(weight_path, by_name=True)

    if train:
        cr = Model([input], [classifier, bbox_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            cr.load_weights(weight_path, by_name=True)
        return cr
    else:
        cr = Model([input], [classifier, bbox_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            cr.load_weights(weight_path, by_name=True)
        return cr


def create_Kao_Pnet( weight_path = 'model12old.h5', train=True):
    if train:
        input = Input(shape = [12,12,3]) # change this shape to [None,None,3] to enable arbitraty shape input
    else:
        input = Input(shape = [None,None,3]) # change this shape to [None,None,3] to enable arbitraty shape input
    conv1 = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(conv1)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    classifier = Conv2D(2, (1, 1), activation='softmax',name='classifier1')(x)
    if train:
        classifier = Reshape((2,), name='cls')(classifier)   # this layer has to be deleted in order to enalbe arbitraty shape input
    bbox_regress = Conv2D(4, (1, 1),name='bbox1')(x)
    if train:
        bbox_regress = Reshape((4,),name='bbox')(bbox_regress) 

    if train:
        cr = Model([input], [classifier, bbox_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            cr.load_weights(weight_path, by_name=True)
        return cr
    else:
        cr = Model([input], [classifier, bbox_regress])
        if os.path.exists(weight_path):
            print('load weights from {}'.format(weight_path))
            cr.load_weights(weight_path, by_name=True)
        return cr

def masked_cls(y_true_full, y_pred):
    y_true = y_true_full[:,:2]
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

def masked_bbox(y_true_full, y_pred):
    y_true = y_true_full[:,2:6]
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return mean_squared_error(y_true * mask, y_pred * mask)

def masked_landmark(y_true_full, y_pred):
    y_true = y_true_full[:,6:]
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return mean_squared_error(y_true * mask, y_pred * mask)

def combine_cls_bbox(ims_cls, one_hot_labels, ims_roi, roi_score):
    ims_all = np.vstack([ims_cls, ims_roi])

    one_hot_labels_ext = np.pad(one_hot_labels, ((0,0),(0,4)), 'constant', constant_values=-1)
    roi_score_ext = np.pad(roi_score, ((0,0),(2,0)), 'constant', constant_values=-1)
    labels_all = np.vstack([one_hot_labels_ext, roi_score_ext])

    return ims_all,labels_all

def combine_cls_bbox_landmark(ims_cls, one_hot_labels, ims_roi, roi_score, ims_pts, pts_score):
    ims_all = np.vstack([ims_cls, ims_roi, ims_pts])

    one_hot_labels_ext = np.pad(one_hot_labels, ((0,0),(0,14)), 'constant', constant_values=-1)
    roi_score_ext = np.pad(roi_score, ((0,0),(2,10)), 'constant', constant_values=-1)
    pts_score_ext = np.pad(pts_score, ((0,0),(6,0)), 'constant', constant_values=-1)
    labels_all = np.vstack([one_hot_labels_ext, roi_score_ext, pts_score_ext])

    return ims_all,labels_all
