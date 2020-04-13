import os
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def detect_red_light(I, kernel):
    '''
    This function takes a numpy array <I> and a numpy array <kernel> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    (n_rows_k,n_cols_k,n_channels_k) = np.shape(kernel) # "k" for "kernel"
    (n_rows,n_cols,n_channels) = np.shape(I)
    
    conv_map = np.zeros((n_rows-n_rows_k+1,n_cols-n_cols_k+1,n_channels))
    kernel_n = np.zeros((n_rows_k,n_cols_k,n_channels_k))
    
    threshold = 0.92
    
    # normalize kernel
    # "ch" for channel
    for ch in range(n_channels_k): 
        norm_k = np.linalg.norm(kernel[:,:,ch])
        kernel_n[:,:,ch] = kernel[:,:,ch]/norm_k
    
    for i in range(n_rows-n_rows_k+1):
        for j in range(n_cols-n_cols_k+1):
            for ch in range(n_channels):
                # normalize cropped image 
                norm_I = np.linalg.norm(I[i:i+n_rows_k,j:j+n_cols_k,ch])
                I_cropped_n = I[i:i+n_rows_k,j:j+n_cols_k,ch]/norm_I
                conv_map[i][j][ch] = np.sum(kernel_n[:,:,ch]*I_cropped_n)
    
    # Weighted combinatioin of RGB channels
    conv_map_rgb = 0.8*conv_map[:,:,0]+0.1*conv_map[:,:,1]+0.1*conv_map[:,:,2]
    
    # Apply threshold
    conv_map_rgb_t = np.where(conv_map_rgb > threshold, conv_map_rgb ,0)
    
    # Find local maximums
    while np.any(conv_map_rgb_t != 0):
        idx = np.where(conv_map_rgb_t == np.amax(conv_map_rgb_t))
        tl_row = int(idx[0])
        tl_col = int(idx[1])
        #print(tl_row,tl_col)
        br_row = tl_row + n_rows_k
        br_col = tl_col + n_cols_k
        bounding_boxes.append([tl_col,tl_row,br_col,br_row])
        
        top = np.max([tl_row-n_rows_k,0])
        bottom = np.min([tl_row+n_rows_k,n_rows-n_rows_k+1])
        left = np.max([tl_col-n_cols_k,0])
        right = np.min([tl_col+n_cols_k,n_cols-n_cols_k+1])
        
        conv_map_rgb_t[top:bottom,left:right] = 0
    
    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
