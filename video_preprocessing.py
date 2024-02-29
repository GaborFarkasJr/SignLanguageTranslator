import cv2
import pandas as pd
import numpy as np
from time import time
from pose_estimation.scripts import pose_api as pose_api
import json

api = pose_api.PoseApi()

# WRIST = 0
# THUMB_CMC = 1
# THUMB_MPC = 2
# THUMB_IP = 3
# THUMB_TIP = 4
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_PIP = 12
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20

hand_landmark_count = 21
face_landmark_count = 468
pose_landmark_count = 33
dim_count = 2 # X, Y

'''
To make the code efficient and easier to read, I will be storing the processed landmarks of the data as a json. Each file will have
a dictionary of features, with the list of values for the x and y landmarks from each frame of the video. For example:

    for the left hand values, the first few arrays would represent [WRIST_X, WRIST_Y, THUMB_CMC_X, THUMB_CMC_Y, ...]. where each is an array of n frames

    * feature_name: name of the feature (
        * left hand = lh
        * right hand = rh
        * face = f
        * pose = p
    )
    * landmark_index: the index of the landmark features (
        * left_hand = 21
        * right_hand = 21
        * face = 468 (on website it says 478, but for some reason this is my limit) 
        * pose = 33
    )
    * landmark_dimension: the dimensionality of each landmark (
        at the moment this is x, y. This can be changed, but the processing will likely take longer and file size will increase.
    )
    
    I added some variables at the top which should make it easier to change to 3d poses instead of 2d
'''
    

def load_video_lookup():
    return pd.read_csv("dataset/wlasl_100_lookup.csv", converters={'video_id': str})

def get_video_data(video_id):
    
    cap = cv2.VideoCapture(f'dataset/WLASL/videos/{video_id}.mp4')
    
    video_data = {
        'left_hand': [[] for _ in range(hand_landmark_count * dim_count)],
        'right_hand': [[] for _ in range(hand_landmark_count * dim_count)],
        'face': [[] for _ in range(face_landmark_count * dim_count)],
        'pose': [[] for _ in range(pose_landmark_count * dim_count)]
    }
    
    success, frame = cap.read()
    while success:
        
        # Converting frame image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face Pose estimation
        face_results = api.face_landmarks_from_image(image)
        for face_index in range(face_landmark_count):
            face = api.face_landmark_from_index(face_results, face_index)
            
            video_data['face'][2*face_index].append(face['x'])
            video_data['face'][2*face_index+1].append(face['y'])
        
        # Pose estimation
        pose_results = api.pose_landmarks_from_image(image)
        for pose_index in range(pose_landmark_count):
            pose = api.pose_landmark_from_index(pose_results, pose_index)
            
            video_data['pose'][2*pose_index].append(pose['x'])
            video_data['pose'][2*pose_index+1].append(pose['y'])
                
        # Hand Pose estimation
        hand_results = api.hand_landmark_from_image(image)
        for hand_index in range(hand_landmark_count):
            left_hand, right_hand = api.hand_landmark_from_index(hand_results, hand_index)
                        
            video_data['left_hand'][2*hand_index].append(left_hand['x'])
            video_data['left_hand'][2*hand_index+1].append(left_hand['y'])
            
            video_data['right_hand'][2*hand_index].append(right_hand['x'])
            video_data['right_hand'][2*hand_index+1].append(right_hand['y'])
            
            
        
        # Adding it to data
        
        
        
        # Doing this last so if the video is finished, it can escape the loop 
        success, frame = cap.read()
        
    return video_data

def get_video_ids(df):
    return df['video_id'].values

if __name__ == "__main__":
    
    video_lookup = load_video_lookup()
    video_ids = get_video_ids(video_lookup)
    
    for video_id in video_ids:
        start_time = time()
        video_data = get_video_data(video_id)
        
        store_data = {
            'lh': video_data['left_hand'],
            'rh': video_data['right_hand'],
            'f': video_data['face'],
            'p': video_data['pose']
        }
        
        with open(f"dataset/processed_spoter_landmarks/{video_id}.json", "w") as outfile:
            json.dump(store_data, outfile)
        
        
            
        duration = time() - start_time
        print(f"{video_id} has been converted in {duration} seconds")
        
    print("Successfully saved")