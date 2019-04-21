import csk
import rgb2gray as rg
import imageio
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
import cv2 # (Optional) OpenCV for drawing bounding boxes

length = 196 # sequence length

# 1st frame's groundtruth information
x1 = 317 # position x of the top-left corner of bounding box
y1 = 140 # position y of the top-left corner of bounding box
width = 105 # the width of bounding box
height = 114 # the height of bounding box

sequence_path = "D://xunlei//bag//" # your sequence path
save_path = "f://img1//" # your save path

tracker = csk.CSK() # CSK instance

for i in range(1,length+1): # repeat for all frames
    frame = imageio.imread(sequence_path+"%08d.jpg"%i)
    if i == 1: # 1st frame
        print(str(i)+"/"+str(length)) # progress
        tracker.init(rg.RGB2GRAY().rgb2gray(frame),x1,y1,width,height) # initialize CSK tracker with GT bounding box
        imageio.imwrite(save_path+'%08d.jpg'%i,cv2.rectangle(frame, \
        (x1, y1), (x1+width, y1+height), (0,255,0), 2)) # draw bounding box and save the frame

    else: # other frames
        print(str(i)+"/"+str(length)) # progress
        x1, y1 = tracker.update(rg.RGB2GRAY().rgb2gray(frame)) # update CSK tracker and output estimated position
        imageio.imwrite(save_path+'%08d.jpg'%i,cv2.rectangle(frame, \
        (x1, y1), (x1+width, y1+height), (0,255,0), 2)) # draw bounding box and save the frame