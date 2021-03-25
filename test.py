#%%
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from PIL import Image
from IPython.display import display

print('module ok')

#%%
class _flagsclass:
    tiny=False
    model = 'yolov4'
    images = ['./data/images/car.jpg']
    iou = 0.45
    score=0.5
    size=416
    weights='./checkpoints/custom-416'
    
FLAGS = _flagsclass()

# %%
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# FLAGS = {
#     'tiny' : False,
#     'model' : 'yolov4',
#     'images' : ['./data/images/car.jpg'],
#     'iou' : 0.45,
#     'score' : 0.5,
#     'size' : 416,
#     'weights' : './checkpoints/custom-416'
# }
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = FLAGS.size
images = FLAGS.images
# %%
saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
print(f'load model ok {FLAGS.weights}')
# %% load image
image_path = images[0]
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
print(original_image.shape)
display( Image.fromarray(original_image) )
image_data = cv2.resize(original_image, (input_size, input_size))
display( Image.fromarray(image_data) )
#정규화 
image_data = image_data / 255.

#%%
image_name = image_path.split('/')[-1]
image_name = image_name.split('.')[0]
print(image_name)
# %%차원추가 
images_data = []
for i in range(1):
    images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)

print(image_data.shape)
print(images_data.shape)
# %% inference
infer = saved_model_loaded.signatures['serving_default']
batch_data = tf.constant(images_data)
pred_bbox = infer(batch_data)
for key, value in pred_bbox.items():
    boxes = value[:, :, 0:4]
    pred_conf = value[:, :, 4:]

#%%
# run non max suppression on detections
boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=50,
    max_total_size=50,
    iou_threshold=FLAGS.iou,
    score_threshold=FLAGS.score
)

# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
original_h, original_w, _ = original_image.shape
bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

detection_num = valid_detections.numpy()[0] 
print(f'vaild detection num : {detection_num}')

#%%
_classes = classes.numpy()[0]
_scores = scores.numpy()[0]
_boxes = boxes.numpy()[0]

pred_r = [[ 
    (
        (_boxes[i][0],_boxes[i][1]),
        (_boxes[i][2],_boxes[i][3]),
        float(_scores[i]),
        int(_classes[i])
    ) 
    for i in range(detection_num)
]]

print(pred_r)

#%%

original_h, original_w, _ = original_image.shape
# bboxes = utils.format_boxes(_boxes, original_h, original_w)

for i in range(detection_num):
    print(f'class : {_classes[i]} , score : {_scores[i]} , {_boxes[i]}')
    #이미지 싸이즈로 조정 
    
    ymin, xmin, ymax, xmax = _boxes[i]
    xmin = xmin * original_w
    ymin = ymin * original_h
    xmax = xmax * original_w
    ymax = ymax * original_h

    print(ymin, xmin, ymax, xmax)

    # crop detection from image (take an additional 5 pixels around all edges)
    cropped_img = original_image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    display( Image.fromarray( cropped_img ) )
    print(cropped_img.shape)

# %%
