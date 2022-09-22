
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from skimage.draw import circle_perimeter, rectangle_perimeter

'''
*Kalman Filter implementation based on:
*        https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py
*        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
*        https://www.kalmanfilter.net/


'''


class KalmanFilter(object):
    
    def __init__(self,init_x, init_v,dt, acceleration_var, x_std, y_std):
        
        self.acceleration_var = acceleration_var

        #init state
        self.x = np.matrix([[0],[0], [0], [0]])
        
        
        #control input variable
        self.u = np.matrix([[init_x], [init_v]])
      
        
        self.dt = dt
        
        
        #state matrix
        self.A = np.matrix([[1, 0 , self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        #control matrix
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt, 0],
                            [0,self.dt]])
        
        
        #transformation matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        # Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * acceleration_var**2
        
        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std**2, 0],
                           [0, y_std**2]])

        #Covariance Matrix
        self.P = np.identity(self.A.shape[1])

        
        
        
    def predict(self):
        
      
        #x_k = A_kx*x+_k-1 + B_k*u_k
        x_new = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        
        # sigma_k = A_k * P+_k-1 * (A_k)^T + Q 
        P_new = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q 
        
        
        self.P = P_new
        self.x = x_new
        
        return self.x[0:2]
        

    
    def update(self,z):
       
        '''
        # y = z - Hx
        
        # error between measurement and prediction
        y = z - (self.H* self.x)
    
       
        # Kalman fain = P* H_t * inv(HPH_t+R)
        
        
        S = self.H* self.P* self.H.T + self.R 
        
        # K = PH'inv(S)
        K = self.P*self.H.T*np.linalg.pinv(S)
        
        
        # x = x + Ky
        x_update = self.x + K*y
        
        # P = (I-KH)P
        P_update = (np.identity(self.H.shape[1]) - K*self.H)*self.P
    
        '''


        S = self.H *self.P *self.H.T +self.R

        # K = PH'inv(S)
        K = self.P * self.H.T * np.linalg.pinv(S)

        # x = x + Ky
        x_update = self.x + K*(z - (self.H*self.x))

        # P = (I-KH)P
        P_update = (np.identity(self.H.shape[1]) - K*self.H)*self.P

    
        self.P = P_update
        self.x = x_update
        
       
        
        return self.x[0]


'''
    alpha - Frame hysteresis for determining active or inactive objects.
    tau - The motion threshold for filtering out noise.
    delta - A distance threshold to determine if an object candidate belongs to an object
            currently being tracked.
    s - The number of frames to skip between detections. The tracker will still work
        well even if it is not updated every frame.
    N - The number of maximum objects to detect.    

'''
class MotionDetector(object):
    def __init__(self,frames, alpha, tau, delta, s, N):

        idx = 2

        #forming a KalmanFilter object
        self.KF =  KalmanFilter(1,1,1,1,0.1,0.1)

        self.frames = frames
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.s = s
        self.N = N

        self.ppframe = rgb2gray(frames[idx-2])
        self.pframe = rgb2gray(frames[idx-1])
        self.cframe = rgb2gray(frames[idx])


    def updateFrame(self, currFrame):

        self.ppframe = rgb2gray(self.frames[currFrame-2])
        self.pframe = rgb2gray(self.frames[currFrame-1])
        self.cframe = rgb2gray(self.frames[currFrame])


        diff1 = np.abs(self.cframe - self.pframe)
        diff2 = np.abs(self.pframe - self.ppframe)

        motion_frame = np.minimum(diff1, diff2)
        thresh_frame = motion_frame > self.tau
        dilated_frame = dilation(thresh_frame, np.ones((9, 9)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)

        collectCenters = []

        for region in regions:
            newVal = self.KF.predict()

            #update filter after predicting
            #converted into list to accces pX and pY values
            #matrix form did not work
            newVal = (self.KF.update(region.centroid)).tolist()

            pX = int(newVal[0][0])
            pY = int(newVal[0][1])

            # add the pX, pY co-ordinates
            collectCenters.append([pX, pY])

        return collectCenters









        
        
  
