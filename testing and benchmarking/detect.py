# -*- coding: utf-8 -*-
import re
import colorsys
import os
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from yolo.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo.utils import letterbox_image
import os
import sys
import argparse



def IoU(a, b, epsilon=1e-5):

    #Intersection
    dx = min(a[3],b[3]) - max(a[1],b[1])
    dy = min(a[2],b[2]) - max(a[0],b[0])
    if (dx >= 0) and (dy >= 0):
        intersection = dx*dy
    else:
        intersection = 0
    w_a = a[3] - a[1]
    w_b = b[3] - b[1]
    h_a = a[2] - a[0]
    h_b = b[2] - b[0]

    union = (np.multiply(w_a,h_a) + np.multiply(w_b, h_b)) - intersection
    iou = intersection / union

    return iou

class YOLO(object):
    _defaults = {
        "model_path": 'SINGLE3424x2432/best.h5',
        "anchors_path": 'model_data/yolo_anchors_6_mc.txt',
        "classes_path": 'model_data/ddsm_class.txt',
        "model_image_size" : (3424, 2432),
        "gpu_num" : 0,
        }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
##################################################################################
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
##################################################################################


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,1)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,1)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=0., iou_threshold=0.)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = image_data[:,:,0]
        image_data = np.expand_dims(image_data, 2)  # Add layers dimension.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print(out_boxes)
        print(out_scores)
        print(out_classes)
        #TODO: Code for highest scored boxes
        std = np.std(out_scores)
        max_score = np.max(out_scores)
        best_scores = list(filter(lambda x: x >= max_score - np.multiply(std,2), out_scores))

        print('Found {} boxes for {}'.format(len(best_scores), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            if out_scores[i] in best_scores:
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{}\n{:.3f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline='white')
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill='red')
                draw.text(text_origin, label, fill='white', font=font)
                del draw

        end = timer()
        print(end - start)
        return image


    def create_output(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = image_data[:,:,0]
        image_data = np.expand_dims(image_data, 2)  # Add layers dimension.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        std = np.std(out_scores)
        max_score = np.max(out_scores)
        best_scores = list(filter(lambda x: x >= max_score - np.multiply(std,2), out_scores))
        pred_boxes = []
        pred_scores = []
        pred_classes = []

        for i in range(len(out_classes)):
            if out_scores[i] in best_scores:
                predicted_class = out_classes[i]
                box = out_boxes[i]
                score = out_scores[i]
                pred_boxes.append(box)
                pred_scores.append(score)
                pred_classes.append(predicted_class)



        return pred_boxes, pred_scores, pred_classes

    def close_session(self):
        self.sess.close()

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def benchmark_YOLO(yolo):
    while True:
        with open(input("Input images text file: ")) as fh:
            lines = fh.read()
        foldername = input("Input folder name for output images: ")
        csvname = input("Input name of benchmark csv: ")
        try:
            os.mkdir(foldername)
        except:
            print("Folder already exists!")
        lineslist = lines.split("\n")
        IoUstack = []
        names = []
        timestack = []
        predclasses = []
        trueclasses = []
        for line in tqdm(lineslist):
            try:
                img, box_cl = line.split(" ")
            except:
                print("It seems like it's something wrong!")
                continue
            if "Normals" in img:
                continue
            try:
                image = Image.open(img)
            except:
                print(img)
                print('Open Error! Passing that one!')
                continue
            pattern = re.compile("([A-Z0-9\.\_]*)\.tif")
            name = pattern.findall(img)[0]
            start = timer()
            pred_box, pred_score, pred_class = yolo.create_output(image)
            box_cl_list = box_cl.split(",")
            xmin = int(box_cl_list[0])
            ymin = int(box_cl_list[1])
            xmax = int(box_cl_list[2])
            ymax = int(box_cl_list[3])
            cl = int(box_cl_list[4])

            box = np.array([ymin, xmin, ymax, xmax])

            IoUlist = []
            classlist = []
            for i in range(len(pred_class)):
                IoUlist.append(IoU(box,pred_box[i]))
                classlist.append(pred_class[i])


                font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                            size=np.floor(3e-4 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 800
                label = 'predicted'.format(pred_class[i])
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = pred_box[i]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                #print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline='red')
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill='red')
                draw.text(text_origin, label, fill='white', font=font)
                #image.save(os.path.join("./outputs",name+"_"+str(i)+".tif"))
                #del draw


            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 800
            label = 'ground truth'.format(cl)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='red')
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill='blue')
            draw.text(text_origin, label, fill='white', font=font)
            image.save(os.path.join(foldername,name+"_benchmark.tif"))
            del draw
            end = timer()
            ev_time = end - start
            idx = IoUlist.index(np.max(IoUlist))
            best_IoU = IoUlist[idx]
            bestIoUclass = classlist[idx]

            IoUstack.append(best_IoU)
            names.append(name)
            timestack.append(ev_time)
            predclasses.append(bestIoUclass)
            trueclasses.append(cl)
        benchmark = pd.DataFrame(
        {'Case': names,
         'IoU': IoUstack,
         'Time': timestack,
         'Predicted class': predclasses,
         'True class': trueclasses
         })
        benchmark.to_csv(os.path.join(foldername,csvname + ".csv"))



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )


    parser.add_argument(
        '--benchmark', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    '''
    Command line positional arguments -- for video detection mode
    '''
    )
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    if FLAGS.benchmark:
        """
        Benchmark mode, disregard any remaining command line arguments
        """
        print("Benchmark mode")
        benchmark_YOLO(YOLO(**vars(FLAGS)))
