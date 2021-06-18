import os
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_pair(images, name, gray=False):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i=0
    
    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i+=1
        
    axes[0].set_title(f'{name}')
    plt.show()
    
def get_poly(ann_path):
    
    with open(ann_path) as handle:
        data = json.load(handle)
    
    shape_dicts = data['shapes']
    
    return shape_dicts
  
def create_binary_masks(im, shape_dicts):
    
    blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
    
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)
        
    return blank

def createDirectory(name):
    # dir that contains the images and annotation json file
    source_dir = os.path.join(img_root, name)
    assert os.path.isdir(source_dir), 'Source dir is not available! Please check if the dir path is correct'
    
    # dir to stor mask
    mask_dir = os.path.join(mask_root, name)
    
    # create the mask directory
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)
    
    return source_dir, mask_dir

def processVideo(name):
    print('------------------------------------------------------------------')
    print(f'Creating masks for video: {name}')
    print('------------------------------------------------------------------')
    
    # make sure all dir are ready to use
    source_dir, mask_dir = createDirectory(name)
    
    # declare some variables
    image_paths = []
    annot_paths = []
    
    for path in os.listdir(source_dir):
        if not path.endswith('.json'):
            image_paths.append(path)
        else:
            annot_paths.append(path)
        
    for image_path in image_paths:
        name = image_path[:image_path.rindex('.')]
        if name + '.json' in annot_paths:
            print(f'Masking for {name}')
            
            image_path = os.path.join(source_dir, image_path)
            img = cv2.imread(image_path, 0)
            
            annot_path = os.path.join(source_dir, name + '.json')
            shape_dicts = get_poly(annot_path)
            
            mask = create_binary_masks(img, shape_dicts)
            cv2.imwrite(os.path.join(mask_dir, name + '_mask.jpg'), mask) 
            
            plot_pair([img, mask], name, gray=True)
            plt.show()
        else:
            print(f'The annot json file for {name} is unavailable. Assumes there is no object.')
            
            image_path = os.path.join(source_dir, image_path)
            img = cv2.imread(image_path, 0)
            
            mask = np.zeros((img.shape[0], img.shape[1]), np.float32)
            cv2.imwrite(os.path.join(mask_dir, name + '_mask.jpg'), mask) 
            
            plot_pair([img, mask], name, gray=True)
            plt.show()
    print()

if __name__=="__main__":
    # video name (default is None, if not metnioned explicitly)
    name = "99 SPEEDMART_TEALIVE.mp4"
    
    # the root folder that stores all the images and annotation
    img_root = r"C:\Users\e-default\Documents\!OpenCV AI Competition\UTAR images" 
    assert os.path.isdir(img_root), 'the path for img_root does not exist'
    
    # the root folder that stores all the masks
    mask_root = r"C:\Users\e-default\Documents\!OpenCV AI Competition\UTAR segmented"
    assert os.path.isdir(mask_root), 'the path for mask_root does not exist'
    
    # if the video name is not explicityly defined
    # we will create masks for extracted iamges from all videos
    if name is None:
        for name in os.listdir(img_root):
            processVideo(name)
    else:
        if '.' in name:
            name = name[:name.rindex('.')]
        processVideo(name)