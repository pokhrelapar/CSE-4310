import numpy as np
from PIL import Image
import skimage.io as io
from numpy.lib import stride_tricks
from change_hsv import rgb_to_hsv
from random import randint, uniform





def random_crop(img,size):
        
    """
    Generates a random square crop of an RGB image 
    Args:
        img - RGB image as numpy array
        size - an integer reflecting the size s.t s->(0,min(w,h))
    Returns:
        img_crop -numpy array of a random square crop of an RGB image
    """
    
    size = int(size/2)
    w,h = img.shape[:2]

    

    limit = min(w,h)
   # print(limit)

        
    rand_x_center = np.random.randint(0,limit)
    #print(rand_x_center)
        
    rand_y_center = np.random.randint(0,limit)
    #print(rand_y_center)

    if 0 <= size <=limit:
        
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
    
        #print(rand_y_center)
    
        # left top x value
        x1 = rand_x_center - size

        # right bottom x val
        x2 = rand_x_center + size
        
        # left top y value
        y1 = rand_y_center - size
       
        
        # right bottom y val
        y2 = rand_y_center + size


        if(x1<0 or x2> w or y1<0  or y2> h):
            print('Random x and y coordinates gave invalid crop size. Run again!')
            
        else:
            img_crop = img[y1:y2,x1:x2]
            #print(type(img_crop))
            #io.imshow(img_crop)
            return img_crop
    else:
            print('Out of bounds')



def extract_patches(img, num_patches):
    
    """
    Creates n^2 non-overlapping patches given an input image, as
    a numpy array, as well as an integer n. interpolation
    Args:
        img - grayscale image
        num_patches - desired number of patches
    Returns:
        patches - 4-D arry of patches
    """
    img_array = np.asarray(img)

    H, W = img_array.shape
        #print(H,W)
    shape = [H// num_patches, W//num_patches]+ [num_patches, num_patches]

    strides = [num_patches * s for s in img_array.strides] + list(img_array.strides)
    #print(type(strides))

    patches = stride_tricks.as_strided(img_array, shape=shape, strides=strides)

        #for i in range(len(patches)):
        #imageio.imwrite('date_set_' + str(i) + '.jpg', patches[i,0,:,:])

    return patches

def resize_img(img, factor):
    
    """
    Resizes an RGB image to a desired scale factor using nearest neighbor interpolation
    Args:
        img - RGB image as numpy array
        factor - an integer representing the desire scale factor
    Returns:
        image_resized - resized image as a numpy array
    """

    #get width and height of original
    h,w= img.shape[:2]


    #scale width and height of original image by a factor = factor
    scaled_w = int(w*factor)
    scaled_h = int(h*factor)



    #numpy array with width = scaled_w and height = scaled_h, and RGB channel
    scaled_image = np.zeros([scaled_h, scaled_w, 3], dtype= np.uint8);

    for i in range(scaled_h):
        for j in range(scaled_w):
            scaled_image[i,j]= img[int(i/factor),int(j/factor)]

    return scaled_image

def color_jitter(img, hue, saturation, value):
    
    """
    Randomly perturbs the HSV values on an input image
    Args:
        img - HSV image as numpy array
        hue = values in [0,360]
        saturation = values in [0,1]
        value = values in [0,1]
    Returns:
       h, s, v channels as images
    """
    
    
 
        
    hsv_image = rgb_to_hsv(img)
    
       
    h = hsv_image[:,:,0] #h
    s = hsv_image[:,:,1] #s
    v = hsv_image[:,:,2] #v
    
    hrand = randint(0,hue)
    srand = uniform(0,saturation)
    vrand = uniform(0,value)
    
    h += hrand
    s += srand
    v += vrand
    
   
    io.imsave('h_channel.jpeg',(h*255.0).astype(np.uint8))
    io.imsave('h_channel.jpeg',(s*255.0).astype(np.uint8))
    io.imsave('h_channel.jpeg',(v*255.0).astype(np.uint8))
    
    
def create_img_pyramid(img,height):
    
    rw,cl,ch = img.shape
    
    form_pyramid = [img] # making a start tuple
  
    
    #need a recursive call : 1/2^(0..h)
    
    for i in range(height-1):
        resized_image = resize_img(img,1/pow(2,i+1)) #resize image recursively
        form_pyramid.append(resized_image)
        
        
    #copied from class's jupyter notebook
    pyramid = tuple(form_pyramid)
    comp_img = np.zeros((rw, cl+cl // 2, ch), dtype=np.double)
    # Place the largest (original) image first
    comp_img[:rw, :cl, :] = pyramid[0]


    
    # Add in the other images
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        comp_img[i_row:i_row + n_rows, cl:cl + n_cols, :] = p
        i_row += n_rows

    #could not save by original image file... so hardcoded
    io.imsave('img.jpeg',pyramid[0])
    
    
    #save images as separate files
    for j in range(height-1):
        image_file = "img_"+ str(2*(j+1)) + "x.jpeg"
        io.imsave(image_file, pyramid[j+1])
        
    #save the whole pyrmaid as an image
    io.imsave('image_pyramid.jpeg',(comp_img).astype(np.uint8))
