import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo.model import preprocess_true_boxes, yolo_body_distributed, yolo_loss, tiny_yolo_body_distributed
from yolo.utils import get_random_data

################################################################################
#Train utils
################################################################################

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes):
    '''create the training model'''
    image_input = Input(shape=(None, None, 1))
    h, w = input_shape
    num_anchors = len(anchors)
    with tf.device('/cpu:0'):
        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body_distributed(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    with tf.device('/cpu:0'):
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes,
             'ignore_thresh': 0.5})([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes):
    '''create the training model, for Tiny YOLOv3'''
    image_input = Input(shape=(None, None, 1))
    h, w = input_shape
    num_anchors = len(anchors)
    with tf.device('/cpu:0'):
        y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body_distributed(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    with tf.device('/cpu:0'):
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes,
             'ignore_thresh': 0.5})([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    return model

################################################################################
#Train script
################################################################################

def _main():
    annotation_path = 'path to annotation file'
    log_dir = 'directory for creating log files'
    classes_path = 'model_data/<classes file>'
    anchors_path = 'model_data/<anchors file>'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    load_pretrained = 'N' #For first epochs

    input_shape = ('height','width') # multiple of 32, hw and 1.7/1 ratio

    is_tiny_version = len(anchors)==6 # default setting

    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes)
    else:
        model = create_model(input_shape, anchors, num_classes)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    test_split = 0.5
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(2137)
    np.random.shuffle(lines)
    np.random.seed(2137)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    num_test = int(num_val * test_split)


    #Stochastic gradient descent used as optimizer
    #To prevent memory allocation failure
    model.compile(optimizer=SGD(lr=1e-5), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    if load_pretrained == "Y":
        model.load_weights(input("Enter path to weights"))
    batch_size = 1

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:num_train+num_test], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=40,
            initial_epoch=0,
            callbacks=[logging, checkpoint])

    model.save_weights(logdir+'weights_stage_1.h5')


################################################################################
#main program
################################################################################
if __name__ == '__main__':
    _main()
