import numpy as np
from PIL import Image
import skimage.io as io
from img_transforms import resize_img




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
