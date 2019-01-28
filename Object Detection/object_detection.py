# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 01:59:42 2019

@author: Muiz Ahmed
"""

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# detect function, frame by frame, passing ssd neural net, images need to be transformed to compatible frame
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    
    # series of transformation is needed
    frame_t = transform(frame)[0]  # right dimensions and right channels
    x = torch.from_numpy(frame_t).permute(2,0,1)  # conversion of numpy array to torch tensors, RBG => GRB for ssd only
    # creating batches of input
    x = Variable(x.unsqueeze(0)) # converting in torch Variable 
    
    # feeding transformed input into neural network
    y = net(x)
    
    # extracting values from tensor
    detections = y.data
    scale = torch.tensor([width, height, width, height])
    
    # elements of detections tensor [batch, number of classes, number of occurance of the class, tuple (score, x0, y0, x1, y1)]
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            j += 1
        
    return frame

# creating SSD neural network
net = build_ssd('test')  # train phase, test phase

# loading pretrained weights
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# creating transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)

writer.close()
    