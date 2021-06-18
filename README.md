# Walkable_Path_Segmentation
Most road surface semantic segmentation convolutional neural network (CNN) models avalailable online are trained from a self-driving car's perspective.
This project is meant to train a CNN for **walkable path road segmentation**, which can deployed in an **OpenCV AI Kit with Depth (OAK-D) device**.

## References:
For image annotation
- [Seth Adams YouTube channel](https://youtu.be/udR6SwojYXo)

For model architecture and training
- [Tensorflow: Image Segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
- [Machine Learning Mastery: Pix2Pix GAN](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/)

Freeze Keras model and saved as pb file
- [Export TensorFlow 2.X Keras model to a frozen and optimized graph](https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb)


## Dependencies
Some open-sourced python packages such as
- numpy
- matplotlib
- cv2
- depthai (if the model is to be deployed in OAK-D or related OAD device)

## Annotation steps
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

## Training the model
Run the training.ipynb in Google Colab
- Change all the directory and file path(s) to the relevant path(s)
- At the end of the notebook, the tensorflow keras model will be freezed and converted into frozen_graph.pb

## Conversion of frozen_graph.pb file to a blob file
Go to [Luxonis Online Tool for Model Conversion](http://luxonis.com:8080/)
- Select **OpenVINO version 2020.4** and **TensorFlow Model**
- For **Model optimizer params**, change to: --data_type=FP16 --mean_values=[127.5,127.5,127.5] --scale_values=[127.5,127.5,127.5] --input_shape=[1,256,256,3]
