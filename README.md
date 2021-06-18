# Walkable_Path_Segmentation
Most road surface semantic segmentation convolutional neural network (CNN) models avalailable online are trained from a self-driving car's perspective.
This project is meant to train a CNN for **walkable path road segmentation**, which can deployed in an **OpenCV AI Kit with Depth (OAK-D) device**.

## Steps
### 0) Download the videos
- Get the videos recorded for training purpose from: https://drive.google.com/drive/folders/1z2OpuTa1k9ORMi4entrGD0FQrFBdGUQi?usp=sharing
- Unzip the file

### 1) Install labelme annotation tool in anaconda
- Open your anaconda prompt
- Install labelme by typing: pip install labelme

### 2) Extract images from video
- go to extract_frames.py
- change variable name into the video name you are assigned to (without all the folder names, just XXX.mp4)
- change variables video_root and img_root into your own folder
- run the code

### 3) Annotate the images
- open anaconda prompt
- type: labelme
- an UI will be opened
- click open dir
- select the dir that contains your extracted images
- click create polygon and start annotate (by clicking obviously)
- after finish annotating, you can click modify polygons to modify the annotation
- type ctrl + s to save this annotation
- DONT change the file name, use default one
- click next image to annotate the next image
- repeat until you annotate all images 

- if tired, can close the UI
- the next time you open the UI and reopen the directory, the annotated images will remain annotated

### 4) Create the masks
- open the code for create_masks.py
- change variable name into the video name you are assigned to (without all the folder names, just XXX.mp4)
- change variables img_root and mask_root
- run the code

IMPORTANT
- refer the image below to see what video_root, img_root and mask_root mean
