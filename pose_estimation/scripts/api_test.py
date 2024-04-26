import cv2
import pose_api
import pprint
import time
import numpy as np


# cap = cv2.VideoCapture(R'C:\Users\dadic\Documents\GitHub\SignLanguageTranslator\bsldict\bsldict\videos_original\k_001_037_000_kick-downstairs.mp4')
cap = cv2.VideoCapture(0) # 'dataset/videos/001_004_004.mp4'
# cap = cv2.VideoCapture('pose_estimation\media\image_test.jpg')
api = pose_api.PoseApi()

start = time.time
ret, frame = cap.read()

winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40, 30)


while ret:    
    
    
    height, width = 244, 244
    b, g, r = 0x00, 0x00, 0x00  # black
    blank = np.zeros((height, width, 3), np.uint8)
    blank[:, :, 0] = b
    blank[:, :, 1] = g
    blank[:, :, 2] = r
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Image is now editable
    image.flags.writeable = True

    
        
    # Resizing image
    # image = cv2.resize(image, (500, 500))
    
    # Facial Feature estimation
    # face_results = api.face_landmarks_from_image(image)
    # api.draw_face_landmarks(blank, face_results)
    
    # # Hand Pose estimation
    hand_results = api.hand_landmark_from_image(image)
    api.draw_hand_landmarks(image, hand_results)
    
    # Pose estimation
    pose_results = api.pose_landmarks_from_image(image)
    api.draw_pose_landmarks(image, pose_results)
    
    
    # Image is now not editable
    image.flags.writeable = False
    
    # reformatting back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    # Flipping image (optional)
    blank = cv2.flip(blank, 1)
    
    
    # # Showing camera view 
    cv2.imshow(winname, image)
    
    ret, frame = cap.read()
    
    # q to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break