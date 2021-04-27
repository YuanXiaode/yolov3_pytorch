import torch
import torchvision.models as models
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import os
import cv2 as cv
capture = cv.VideoCapture("/home/yxd/mygithub/yolov5/runs/detect/导线舞动素材5.mp4")
cv.namedWindow("video",cv.WINDOW_AUTOSIZE)
while(1):
    ret, frame = capture.read()
    if ret:
        cv.imshow(frame)
        key = cv.waitKey(20)
        if key==ord('q'):
            break;
    else:
        break
capture.release()
cv.destroyAllWindows()
