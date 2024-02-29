import cv2
import pose_api
import pprint
import time
import numpy as np


# cap = cv2.VideoCapture(R'C:\Users\dadic\Documents\GitHub\SignLanguageTranslator\bsldict\bsldict\videos_original\k_001_037_000_kick-downstairs.mp4')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('pose_estimation\media\image_test.jpg')
api = pose_api.PoseApi()



start = time.time
ret, frame = cap.read()
while ret:    
    
        
    # Converting frame image from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    
    
    # Image is now editable
    image.flags.writeable = True
    
    # Resizing image
    # image = cv2.resize(image, (500, 500))
    
    # # Facial Feature estimation
    face_results = api.face_landmarks_from_image(image)
    api.draw_face_landmarks(image, face_results)
    print(np.shape(api.get_face_landmark_coordinates(face_results)))
    
    # # Hand Pose estimation
    # hand_results = api.hand_landmark_from_image(image)
    # api.draw_hand_landmarks(image, hand_results)
    
    # Pose estimation
    # pose_results = api.pose_landmarks_from_image(image)
    # api.draw_pose_landmarks(image, pose_results)
    # pprint.pprint(api.get_pose_landmark_coordinates(pose_results))
    
    
    # Image is now not editable
    image.flags.writeable = False
    
    # reformatting back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    # Flipping image (optional)
    image = cv2.flip(image, 1)
    
    # # Showing camera view 
    cv2.imshow('frame', image)
    
    ret, frame = cap.read()
    
    # q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break