import sys
import random
import argparse
import os
import traceback
from PySide2 import QtCore, QtWidgets, QtGui
from skvideo.io import vread

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from skimage.draw import circle_perimeter
from skimage import data, draw, io

from kalman import KalmanFilter, MotionDetector

#random colors
r = random.randint(0,255)
g = random.randint(0,255)
b = random.randint(0,255)

#used disk to form trails of tracked object, bounding box not used 
class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames

        self.current_frame = 0

        self.button2 = QtWidgets.QPushButton("Next Frame @ 60s")
        self.button3 = QtWidgets.QPushButton("Previous Frame @ 60s")

        self.history = []
        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.button2.clicked.connect(self.on_click2)
        self.button3.clicked.connect(self.on_click3)
        self.frame_slider.sliderMoved.connect(self.on_move)


    @QtCore.Slot()
    def on_click2(self):
    
        collectCenters = []

        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        collectCenters = motionObject.updateFrame(self.current_frame)

        self.history.append(collectCenters)
        

        #fast forward 60s into the next frame
        self.current_frame +=60
        print('Showing frame :', end="") 
        print(self.current_frame)
        

        #https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.disk
        for i in range(len(self.history)):
            for j in range(len(self.history[i])):
                center = (int(self.history[i][j][0]), int(self.history[i][j][1]))
                #print(center)
                rr, cc = draw.disk(center, 4)
                #print(rr,cc)
                #print( self.frames[self.current_frame][rr, cc])
                self.frames[self.current_frame][rr, cc] = (178, 34, 34 )
           

    @QtCore.Slot()
    def on_click3(self):
    
        #reinitilaize for previous frames to clear history
        collectCenters = []
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))


        self.history = []
        collectCenters = motionObject.updateFrame(self.current_frame)

        self.history.append(collectCenters)

        #fast forward 60s into the previous frame
        self.current_frame -=60

        print('Showing frame :', end="") 
        print(self.current_frame)
        

        #https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.disk
        for i in range(len(self.history)):
            for j in range(len(self.history[i])):
                center = (int(self.history[i][j][0]), int(self.history[i][j][1]))
                #print(center)
                rr, cc = draw.disk(center, 4)
                self.frames[self.current_frame][rr, cc] = (178, 34, 34 )

    @QtCore.Slot()
    def on_move(self, pos):
        self.current_frame = pos
        print('Showing frame :', end="") 
        print(self.current_frame)
        
        #clear history to re-initialzie frames
        self.history.clear()
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    #Forming a MotionDetector object
    motionObject = MotionDetector(frames,0.1,0.2,0.1, 0.2,2)
    app = QtWidgets.QApplication([])

    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())

    