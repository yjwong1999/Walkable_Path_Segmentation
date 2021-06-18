import cv2
import depthai as dai
import numpy as np
#import timeit

nnPathDefault = r'C:\Users\e-default\Documents\!OpenCV AI Competition\frozen_graph_new.blob'

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Create color cam node
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(256, 256)
camRgb.setInterleaved(False)
camRgb.setFps(10)

# create image manip node
manip = pipeline.createImageManip()
manip.initialConfig.setResize(256, 256)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)

# create nn node
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(nnPathDefault)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# link the nodes
camRgb.preview.link(manip.inputImage)
manip.out.link(nn.input)

# output nodes
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qNet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    def customReshapeV1(x, target_shape):
      row, col, ch = target_shape
      arr3d = []
      arr2d = None
    
      for i in range(len(x)//col):
          if i % col == 0 and i != 0:
              arr3d.append(arr2d)
              arr2d = None
      
          idx1 = i * col
          idx2 = idx1 + col
          arr1d = np.reshape(x[idx1:idx2], (1, col, 1))
      
          if arr2d is None:
              arr2d = arr1d.copy()
          else:
              arr2d = np.concatenate((arr2d, arr1d), axis = 0)
        
      arr3d.append(arr2d)
      arr3d = np.concatenate(arr3d, axis=-1)
    
      return arr3d
    
    def customReshape(x, target_shape):
      x = np.reshape(x, target_shape, order='F')
      for i in range(3):
          x[:,:,i] = np.transpose(x[:,:,i])
    
      return x
    
    def show_deeplabv3p(output_colors, mask):
        mask = ((mask + 1) / 2 * 255).astype(np.uint8)
        return cv2.addWeighted(mask,0.8, output_colors,0.5,0)
    
    t1 = 0
    t2 = 0
    # start looping
    while True:
        # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
        inRGB = qRgb.tryGet()
        inNet = qNet.tryGet()

        if inRGB is not None:
            rgb = inRGB.getCvFrame()
            cv2.imshow('rgb', rgb)
            
        if inNet is not None:
            '''
            t2 = timeit.default_timer()
            print(t2-t1, t1, t2)
            t1 = t2
            '''
            mask = inNet.getFirstLayerFp16()
            mask = np.array(mask)
            mask = customReshape(mask, (256, 256, 3))
            mask = show_deeplabv3p(rgb, mask)
            cv2.imshow('mask', mask)

        # quit if user pressed 'q'
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows() 
            break