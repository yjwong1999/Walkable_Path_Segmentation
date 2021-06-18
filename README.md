# Walkable_Path_Segmentation
Most road surface semantic segmentation convolutional neural network (CNN) models avalailable online are trained from a self-driving car's perspective.
This project is meant to train a CNN for **walkable path road segmentation**, which can deployed in an **OpenCV AI Kit with Depth (OAK-D) device**.

## Dependencies
Some open-sourced python packages such as
- numpy
- matplotlib
- cv2
- depthai (if the model is to be deployed in OAK-D or related OAD device)

## Steps
### 0) Download the videos
- Get the videos recorded for training purpose from: https://drive.google.com/drive/folders/1z2OpuTa1k9ORMi4entrGD0FQrFBdGUQi?usp=sharing
- Unzip the file

### 1) Install labelme annotation tool in anaconda
- Open your anaconda prompt
- Install labelme by typing: pip install labelme

### 2) Extract images from video
- Open extract_frames.py
- Change variable name into the video name you are assigned to (without all the folder names, just XXX.mp4)
- Change variables video_root and img_root into your own folder
- Run the code

### 3) Annotate the images
- Open anaconda prompt
- Launch labelme by typing: labelme
- Click open dir to select the dir that contains your extracted images
- Click create polygon and start annotating
- Save your annotation using the default filename and file location (don't change anything)
- Repeat until you annotate all images for all videos you have

### 4) Create the masks
- Open the code for create_masks.py
- Change variable name into the video name you are assigned to (without all the folder names, just XXX.mp4)
- Change variables img_root and mask_root
- Run the code

IMPORTANT
- refer the image below to see what video_root, img_root and mask_root mean
