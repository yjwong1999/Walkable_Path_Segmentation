import cv2
import os

def createDirectory(name):
      # Video path
      video_path = os.path.join(video_root, name)
      assert os.path.isfile(video_path), 'Video is not available! Please check if the video path is correct'

      # Dir to store image
      dest_folder = os.path.join(img_root, name[:name.rindex('.')])
  
      # create the destination directory
      if not os.path.isdir(dest_folder):
          os.mkdir(dest_folder)
          
      return video_path, dest_folder
        
def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    # success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*3000))    # added this line 
        success,image = vidcap.read()
        print (f'Read frame {count}: ', success)
        if success:
            cv2.imwrite( pathOut + f"\\frame{count}.jpg", image)     # save frame as JPEG file
            count = count + 1
        else:
            break

def processVideo(name):
    print('------------------------------------------------------------------')
    print(f'Extract images from video: {name}')
    print('------------------------------------------------------------------')
    video_path, dest_folder = createDirectory(name)
    extractImages(video_path, dest_folder)
    print()
    
if __name__=="__main__":
    # video name (default is None, if not mentioned explicitly)
    name = "99 SPEEDMART_TEALIVE.mp4"
    
    # root folder for video
    video_root = r"C:\Users\e-default\Documents\!OpenCV AI Competition\UTAR videos ML\UTAR videos ML"
    assert os.path.isdir(video_root), 'the path for video_root does not exist'
    
    # root folder for all frames
    img_root = r"C:\Users\e-default\Documents\!OpenCV AI Competition\UTAR images" 
    assert os.path.isdir(img_root), 'the path for img_root does not exist'

    # if the video name is not explicitly defined
    # we will extract images from all videos
    if name is None:
        for name in os.listdir(video_root):
            processVideo(name)
    else:
        processVideo(name)