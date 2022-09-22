import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from PIL import Image
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from skimage.feature import SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp, rotate 
from skimage import measure, transform, data


def match_descriptors(descriptor1, descriptor2,cross_check=True):
    
    distances = cdist(descriptor1, descriptor2, 'euclidean')

    idx1 = np.arange(descriptor1.shape[0])

    idx2 = np.argmin(distances, axis=1)
    
  
    #https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/feature/match.py#L5-L97
    
    '''
    Alterate:
                calculate distance = cdist(descriptor2, descriptor1, 'euclidean')
                 idx1 = np.arange(descriptor1.shape[0])
                 idx2 = np.argmin(distances, axis=1)
                 
                 use a for loop to check where they match
                
    
    
    '''
    if cross_check:
        match1 = np.argmin(distances, axis=0)
        #print('match')
        #print(match1)
        
        
        bool_mask = idx1 == match1[idx2]
        #print('match')
        #print(mask)
        
        idx1 = idx1[bool_mask]
        #print('idx1')
        #print(idx1)
        
        idx2 = idx2[bool_mask]
        #print('idx2')
        #print(idx2)

  
    matches = np.column_stack((idx1,idx2))
    
    #print('matches')
  
    

    return matches





def compare_images(first_image, second_image):
    
    final_shape1 = list(first_image.shape)

    final_shape2 = list(second_image.shape)



    shape1 = first_image.shape[0]

    shape2 = second_image.shape[0]



    shape11 = first_image.shape[1]

    shape22 = second_image.shape[1]
    

    
    if shape1 < shape2 :
        final_shape1[0] = shape2

    elif shape1 > shape2 :
        final_shape2[0] = shape1


    if shape11 < shape22:
        final_shape1[1] = shape22

    elif shape11 > shape22:
        final_shape2[1] = shape11

    
    
    if final_shape1 != first_image.shape:
        final_image1 = np.zeros(final_shape1, dtype=first_image.dtype)

        final_image1[:first_image.shape[0], :first_image.shape[1]] = first_image

        first_image = final_image1

    if final_shape2 != second_image.shape:
        final_image2 = np.zeros(final_shape2, dtype=second_image.dtype)
        final_image2[:second_image.shape[0], :second_image.shape[1]] = second_image
        second_image = final_image2
        
    return first_image, second_image

    
        
#Implementatin for plot_matches based on :
#https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/util.py#L43-L138


def plot_matches1(image1, image2, keypoints1, keypoints2, matches):

    image1, image2 = compare_images(image1, image2)
    
    '''
    h1,w1 = image1.shape[:2]
    print(w1,h1)
    
    new_h1, new_w1 = ((2*h1),(2*w1))
    print(new_w1, new_h1)
    
    h2,w2 = image2.shape[:2]
    print(w2,h2)
    
    new_h2, new_w2 = ((2*h2),(2*w2))
    print(new_w2, new_h2)
   
    '''
    
  

    
    stack_image = np.array(image1.shape)

    #join across horizontal axis, image is the joint image of 1 and 2
    image = np.concatenate([image1,image2], axis=1)
    
    stack_image[0] = 0

    
    # Gives a better matching between plots
    fig, ax  = plt.subplots(1,1,figsize=(10,5))
  


    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    ax.scatter(keypoints1[:, 1], keypoints1[:, 0], facecolors='aqua', edgecolors='k')
    ax.scatter(keypoints2[:, 1] + stack_image[1], keypoints2[:, 0] + stack_image[0], facecolors='maroon', edgecolors='k')



    ax.imshow(image, cmap='gray')
    ax.axis((0,image1.shape[1] + stack_image[1], image1.shape[0]+ stack_image[0],0))
    ax.axis('off')

    for i in range(matches.shape[0]):

        idx1 = matches[i,0]
        idx2 = matches [i,1]

        ax.plot((keypoints1[idx1,1],keypoints2[idx2,1] + stack_image[1]),(keypoints1[idx1,0],keypoints2[idx2,0] + stack_image[0]))
    plt.title('plot matches')
    plt.show()   


def compute_affine_transform(keypoints1,keypoints2):

   
    
    N = len(keypoints1)
    
    matrix_x    = []
    matrix_xcap = []
    
    
    
    new_size = (2*len(keypoints1),1)
    affine_size = (2,3)
    affine_row = [0, 0, 1]
    
    
    #https://dillhoffaj.utasites.cloud/posts/random_sample_consensus/
    for j in range(N):
        matrix_x.append([keypoints1[j][0], keypoints1[j][1], 1, 0, 0, 0])
        matrix_x.append([0, 0, 0, keypoints1[j][0], keypoints1[j][1],1])
        matrix_xcap.append(keypoints2[j][0])
        matrix_xcap.append(keypoints2[j][1])
        
    like_x = np.array(matrix_x)
    
    matrix_xcap_new = np.reshape(matrix_xcap,new_size)
    
    like_x_transpose = np.transpose(like_x)
    
    product1 = np.dot(like_x_transpose,like_x)
    
    product1_inverse = np.linalg.inv(product1)
    
    product2 = np.dot(like_x_transpose,matrix_xcap_new)
    
    
    #Normal equation : A' Ax = A' 
    
    #Take inverse
    
    affine_matrix = np.dot(product1_inverse, product2)
    affine_matrix = np.reshape(affine_matrix,affine_size)
    affine_matrix = np.r_[affine_matrix, [affine_row]]
    #print(affine_matrix)
    
    return affine_matrix




def compute_projective_transform(keypoints1, keypoints2):
    
    N = len(keypoints1)
    
    matrix_x1   = []
    matrix_xcap1 = []
    
    
    
    new_size = (2*len(keypoints1),1)
    projective_size = (3,3)
    projective_row = [1]
    
    
    #https://dillhoffaj.utasites.cloud/posts/random_sample_consensus/
    for j in range(N):
        matrix_x1.append([keypoints1[j][0], keypoints1[j][1], 1, 0, 0, 0, -(keypoints1[j][0] * keypoints2[j][0]), -(keypoints1[j][1] * keypoints2[j][0])])
        matrix_x1.append([0, 0, 0, keypoints1[j][0], keypoints1[j][1],1, -(keypoints1[j][0] * keypoints2[j][1]), -(keypoints1[j][1] * keypoints2[j][1])])
        matrix_xcap1.append(keypoints2[j][0])
        matrix_xcap1.append(keypoints2[j][1])
   
    
    
    
        
    like_x1 = np.array(matrix_x1)

    
    matrix_xcap_new1 = np.reshape(matrix_xcap1,new_size)
    
    like_x_transpose = np.transpose(like_x1)
    
    product1 = np.dot(like_x_transpose,like_x1)
    
    product1_inverse = np.linalg.inv(product1)
    
    product2 = np.dot(like_x_transpose,matrix_xcap_new1)
    
    
    #Normal equation : A' Ax = A' 
    
    #Take inverse
    
    projective_matrix = np.dot(product1_inverse, product2)
 
    projective_matrix = np.r_[projective_matrix, [projective_row]]
    
    projective_matrix = np.reshape(projective_matrix,projective_size)
    
    
    return projective_matrix
  


from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp, AffineTransform

#Implementation based on : https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/fit.py#L625-L899

def ransac(data, model_class, min_samples,threshold, max_iterations):
    
    
    num_inliers = 0 
    bestErr = np.inf 
    best_inliers = []
    bestFit = None
    
    bestFit = np.random.default_rng()
    #print(bestFit) 
    
    num_samples = len(data[0])
    
    model = model_class()
    
    iterations = 0
    
    while iterations < max_iterations: 
        
        
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        
        idxs_rnd = bestFit.choice(num_samples, min_samples, replace=False)
   
        
        samples = [d[idxs_rnd] for d in data]
        
        success = model.estimate(*samples)
        
        errors = np.abs(model.residuals(*data))
        
        inliers = errors < threshold
        
        sum_of_errors = errors.dot(errors)
        
        total_inliers = np.count_nonzero(inliers)
        
        iterations += 1 
        
        if (total_inliers > num_inliers or (total_inliers == num_inliers and sum_of_errors < bestErr)):
            num_inliers = total_inliers
            bestErr = sum_of_errors
            best_inliers = inliers
            

    if any(best_inliers):
            
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        
    return model, best_inliers


def main():

    # change paths as required
    dst_image_rgb  = np.asarray(Image.open(r"C:\Users\apu84\Downloads\a2_images\Rainier1.png"))

    src_image_rgb  = np.asarray(Image.open(r"C:\Users\apu84\Downloads\a2_images\Rainier2.png"))

        

    if dst_image_rgb.shape[2]== 4:
            dst_image_rgb = rgba2rgb(dst_image_rgb)

    if src_image_rgb.shape[2]== 4:
            src_image_rgb = rgba2rgb(src_image_rgb)


    dst_image = rgb2gray(dst_image_rgb)
    src_image = rgb2gray(src_image_rgb)




        
    detector1 = SIFT()
    #print(type(detector1))
    detector2 = SIFT()
    #print(type(detector1))
    #detector3 = SIFT()

    detector1.detect_and_extract(dst_image)
    detector2.detect_and_extract(src_image)

    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors



    matches12 = match_descriptors(descriptors1, descriptors2,cross_check=True)
    dst = keypoints1[matches12[:, 0]]
    src = keypoints2[matches12[:, 1]]

    matches12 = match_descriptors(descriptors1, descriptors2,cross_check=True)
    dst = keypoints1[matches12[:, 0]]
    src = keypoints2[matches12[:, 1]]

    print("Close each of the plot to see the result")
    print("First plot shows the result of plot_matches1")
    print("Second plot shows the result of ransac")
    print("Third plot shows the result of stitiching")

    plot_matches1(dst_image, src_image, keypoints1, keypoints2, matches12)


        # ProjectiveTransform AffineTransform
    sk_M, sk_best = ransac((src[:, ::-1], dst[:, ::-1]), ProjectiveTransform,4, 1,300)

    #print(type(sk_M))
    #print(np.count_nonzero(sk_best))
    src_best = keypoints2[matches12[sk_best, 1]][:, ::-1]
    dst_best = keypoints1[matches12[sk_best, 0]][:, ::-1]

    fig = plt.figure(figsize=(8, 4))
    plt.title('running ransac')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    ax1.imshow(dst_image_rgb)
    ax2.imshow(src_image_rgb)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
        ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')
    
    plt.show()

    #class's jupyter notebook

    # Transform the corners of img1 by the inverse of the best fit model
    rows, cols = dst_image.shape
    corners = np.array([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])

    corners_proj = sk_M(corners)
    all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    #print(output_shape)

    offset = SimilarityTransform(translation=-corner_min)
    dst_warped = warp(dst_image_rgb, offset.inverse, output_shape=output_shape)

    tf_img = warp(src_image_rgb, (sk_M + offset).inverse, output_shape=output_shape)

    # Combine the images
    foreground_pixels = tf_img[tf_img > 0]
    dst_warped[tf_img > 0] = tf_img[tf_img > 0]

    # Plot the result

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(dst_warped)

    plt.title('after stitching')
    plt.show()

        


if __name__ == "__main__":
    main()



'''

#could not get my computed matrix to work with the ransac function, so using built in


'''
