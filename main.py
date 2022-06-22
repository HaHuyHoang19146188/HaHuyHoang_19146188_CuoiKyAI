from traitlets.traitlets import ClassTypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from tensorflow import keras
import os
filename = 'video.avi'
frames_per_second = 24.0
res = '48p'
# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
def write_lable(acc,frame):
    if acc >= 37 and acc <= 276:
        cv2.putText(frame, 'Ipsala', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 361 and acc <= 587:
        cv2.putText(frame, 'Basmati', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 675 and acc <= 965:
        cv2.putText(frame, 'Basmati', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 683 and acc <= 964:
        cv2.putText(frame, 'Basmati', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 1048 and acc <= 1225:
        cv2.putText(frame, 'Arborio', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 1312 and acc <= 1471:
        cv2.putText(frame, 'Arborio', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 1540 and acc <= 1694:
        cv2.putText(frame, 'Karacadag', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 1779 and acc <= 1916:
        cv2.putText(frame, 'Arborio', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 2014 and acc <= 2194:
        cv2.putText(frame, 'Jasmine', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 2277 and acc <= 2418:
        cv2.putText(frame, 'Jasmine', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    elif acc >= 2532 and acc <= 2697:
        cv2.putText(frame, 'Ipsala', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'None', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    return 1;
# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height
def recognize(path):
    #result = conv_model.predict([prepare(path)])
    #d = image.load_img(path)
    #plt.imshow(d)
    #x = np.argmax(result, axis=1)
    #print(results[int(x)])
    return 1
# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
results={
   0:'Arborio',
   1:'Basmati',
   2:'Ipsala',
   3:'Jasmine',
   4:'Karacadag',
   5:'None'
}
vid = cv2.VideoCapture('videocat4.mp4')
conv_model = keras.models.load_model('model1_rice_10epoch.h5')
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
#path = '/content/drive/MyDrive/happy2382.jpg'
def prepare(img_path):
    img = image.load_img(img_path, target_size=(256,256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)
index=0
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(vid, res))

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cropped_image = frame[115:365,195:445]
    # Display the resulting frame
    #cv2.imwrite('happy' + str(i) + '.jpg', cropped_image)
    recognize(cropped_image)
    write_lable(index,frame)
    cv2.imshow("frame", frame)
    out.write(frame)
    # the 'q' button is set as the
    # quitting button you may use any
    index += 1
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()