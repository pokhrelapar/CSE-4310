"""
 * @author [Apar Pokhrel;]
 * @email [apar.pokhrel@mavs.uta.edu]
 * @create date 2022-02-14 14:44:55
 * @modify date 2022-02-14 14:44:55
 * @desc [description]
"""


import numpy as np
import math
import skimage.io as io
from PIL import Image
from matplotlib import pyplot as plt


import sys




def validate_args(args):
    """
    Checks if the first argument in the list is a number.
    Args:
        args (list) - Command line arguments
    Returns:
        0 if the first argument is a number
        1 if the first argument is not a number
        2 if the list is empty
    """
    while(len(args) <5):
        print('Usage xxx.py -imagepath -hue -saturation -value')
        sys.exit(-1)



def rgb_to_hsv(rgb):
    

    #rgb is a numpy array     
    rw = rgb.shape[0]
    cl= rgb.shape[1]
    #print(rw,cl)

    #Changing RGB range value from [0,255] to [0,1]
    rgbNew = rgb / 255.0
    
    #print(type(rgbNew))
    #print(rgbNew)


    r= rgbNew[:,:,0]
    #print(r)
    g=rgbNew[:,:,1]
    #print(g)
    b=rgbNew[:,:,2]
    #print(b)


    maxVal = np.max(rgbNew, axis=2)
    #print(type(maxVal))
    value = maxVal
    minVal = np.min(rgbNew, axis=2)
    #print(type(minVal))
    
    
    chromaVal = maxVal - minVal
    #print(type(chromaVal))
    hueVal= np.zeros([rw,cl])
    saturation = np.zeros([rw,cl])

    for i in range(0,rw-1):
        for j in range(0,cl-1):
            if value[i][j] == 0:      # s=0 if v==0 else c/v
                 saturation[i][j] = 0
            else:
                saturation[i][j]= chromaVal[i][j]/maxVal[i][j]
                

    for i in range(0,rw-1):
        for j in range(0,cl-1):
            if chromaVal[i][j]== 0:
                hueVal[i][j]= 0

            elif maxVal[i][j]== r[i][j]:
                hueVal[i][j] =  math.fmod(((g[i][j]-b[i][j])/chromaVal[i][j]),6)

            elif maxVal[i][j]== g[i][j]:
                hueVal[i][j] = (((b[i][j]-r[i][j])/chromaVal[i][j]) + 2)

            elif maxVal[i][j]== b[i][j]:
                hueVal[i][j] = (((r[i][j]- g[i][j])/(chromaVal[i][j])) + 4)
    
    hue = (hueVal*60.0)      
    
    #https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    hsv = np.dstack([hue,saturation,value])
    #print(type(hsv))

    return hsv

def hsv_to_rgb(hsv):
    
    hue = hsv[:,:,0] #h
    saturation = hsv[:,:,1] #s
    value = hsv[:,:,2] #v
    
    rw,cl = hsv.shape[:2]
    
    chromaVal =  np.zeros([rw,cl])
    X =  np.zeros([rw,cl])
    R = np.zeros([rw,cl])
    G = np.zeros([rw,cl])
    B = np.zeros([rw,cl])
    r = np.zeros([rw,cl])
    g = np.zeros([rw,cl])
    b = np.zeros([rw,cl])
    
  
    
   # = s * v

    for i in range(0, rw-1):
        for j in range(0, cl-1):
            chromaVal[i][j] = value[i][j]*saturation[i][j]
            
    
    
  
    hue/= 60.0
    
    for i in range(0, rw-1):
        for j in range(0, cl-1):
             X[i][j] = chromaVal[i][j] *  (1 - abs(math.fmod(hue[i,j],2) - 1))
    
 
    for i in range(0, rw-1):
        for j in range(0, cl-1):
            if 0 <= hue[i][j] and  hue[i][j] < 1:
                r[i][j] = chromaVal[i][j]
                g[i][j] =  X[i][j]
                b[i][j] =  0
                
                
            elif 1 <= hue[i][j] and  hue[i][j] < 2:
                r[i][j] = X[i][j]
                g[i][j] = chromaVal[i][j]
                b[i][j] = 0
                
                
            elif 2 <= hue[i][j] and  hue[i][j] < 3:
                r[i][j] = 0
                g[i][j] = chromaVal[i][j]
                b[i][j] = X[i][j]
                
                
            elif 3 <= hue[i][j] and  hue[i][j] < 4:
                r[i][j] = 0
                g[i][j] = X[i][j]
                b[i][j] = chromaVal[i][j]
                
                
            elif 4 <= hue[i][j] and  hue[i][j] < 5:
                r[i][j] = X[i][j]
                g[i][j] = 0
                b[i][j] = chromaVal[i][j]
                
            elif 5 <= hue[i][j] and  hue[i][j] < 6:
                r[i][j] = chromaVal[i][j]
                g[i][j] = 0
                b[i][j] = X[i][j]
                
            
    m = value - chromaVal 
            
    R = (r+m)
    G = (g+m)
    B = (b+m)
    
    RGB = np.dstack([R,G,B])
    
    return RGB
    
    


def main(args):
    """
    The main function of our program.
    """

    num_args = len(args) # `args` is a list object, `len` returns the length

    validate_args(args)


    
    # Attempt to convert the first argument to an integer
    try:
    
        int(args[2])
        float(args[3])
        float(args[4])
    
    except ValueError:
        print('h,s,v is not a number ')

    
    hue = int(args[2])
    saturation = float(args[3])
    value = float(args[4])


    
    if int(args[2]) <0.0 or int(args[2]) >360.0 :
        print('h must be between [0,360]')
        sys.exit(-1)


    elif float(args[3]) <0.0 or float(args[3]) >1.0:
        print('s must be between [0,1]')
        sys.exit(-1)

    elif float(args[4]) <0.0 or float(args[4]) >1.0:
        print('v must be between [0,1]')
        sys.exit(-1)
    



    rgb = io.imread(args[1])
    rgb_array = np.asarray(rgb)
    rw, cl, ch = rgb_array.shape[:3]

    #hsv_array = np.zeros((rw,cl,ch))
    rgb_image_array = np.zeros((rw,cl,ch))


    #change rgb image array to hsv image array
    hsv_array = rgb_to_hsv(rgb_array)

    #extract h,s,v values from the hsv image array
    hsv_array[:,:,0] += hue
    hsv_array[:,:,1] += saturation
    hsv_array[:,:,2] += value


    rgb_image_array = hsv_to_rgb(hsv_array)
    print('Saving modified image')
    io.imsave('modified_image.jpeg',(rgb_image_array*255.0).astype(np.uint8))
    #print(type(hsv_image))


if __name__ == "__main__":
    main(sys.argv)
