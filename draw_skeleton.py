import cv2
import numpy as np 
from pose_estimation.scripts import pose_api as pose_api
api = pose_api.PoseApi()

def draw_skeleton(file_name):
    '''
    Returns video in the shape of [frames, height, width, channels]
    '''
    
    height, width = 244, 244 # Popular size for resnet
    b, g, r = 0x00, 0x00, 0x00  # black background
    processed_video = []
    
    cap = cv2.VideoCapture(f'dataset/videos/{file_name}.mp4')
    
    success, frame = cap.read()
    while success:
        processed_frame = np.zeros((height, width, 3), np.uint8)
        processed_frame[:, :, 0] = b
        processed_frame[:, :, 1] = g
        processed_frame[:, :, 2] = r
        
        # Converting frame image from BGR to RGB
        video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # Image is now editable
        video_frame.flags.writeable = True
        
        # Converting frame image from BGR to RGB
        
        video_frame[:, :, 1] = 25
        
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY)
        
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2RGB)
        
        video_frame[:, :, 0] += 40
        video_frame[:, :, 1] += 10
        video_frame[:, :, 2] += 10
        
        # Facial Feature estimation
        # face_results = api.face_landmarks_from_image(video_frame)
        # api.draw_face_landmarks(processed_frame, face_results)
        
        # # Hand Pose estimation
        hand_results = api.hand_landmark_from_image(video_frame)
        api.draw_hand_landmarks(processed_frame, hand_results)
        
        # Pose estimation
        pose_results = api.pose_landmarks_from_image(video_frame)
        api.draw_pose_landmarks(processed_frame, pose_results)
        
        
        processed_video.append(processed_frame)
        
        # Doing this last so if the video is finished, it can escape the loop 
        success, frame = cap.read()
        
    processed_video = np.array(processed_video)
    
    return processed_video

