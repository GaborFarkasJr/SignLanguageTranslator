import cv2
import numpy as np 
from pose_estimation.scripts import pose_api as pose_api
import json
api = pose_api.PoseApi()

def video_to_hand_landmarks(file_name):
    hand_landmarks = []
    cap = cv2.VideoCapture(f'dataset/WLASL/videos/{file_name}.mp4')
    
    success, frame = cap.read()
    while success:
        
        # Converting frame image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        # Hand Pose estimation
        hand_results = api.hand_landmark_from_image(image)
            
        # Getting landmark coordinates
        lh, rh = api.get_hand_landmark_coordinates(hand_results)
        
        # Only adds landmarks if hands are detected
        if lh != None or rh != None: hand_landmarks.append(lh + rh)
        
         # Doing this last so if the video is finished, it can escape the loop 
        success, frame = cap.read() 
    
    
    return hand_landmarks

# def video_to_face_landmarks(file_name):
#     cap = cv2.VideoCapture(f'bsldict/bsldict/videos_original/{file_name}')
#     face_landmark_list = []
    
#     success, frame = cap.read()
#     while success:

#         # Converting frame image from BGR to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
#         # Face pose estimation
#         face_results = api.face_landmarks_from_image(image)
#         face_landmark_list.append(api.get_face_landmark_coordinates(face_results))
        
#         # Doing this last so if the video is finished, it can escape the loop 
#         success, frame = cap.read()  
          
#     return face_landmark_list

def landmark_from_json(video_id):
    landmark_data = {}
    with open(f"dataset/processed_spoter_landmarks/{video_id}.json", "r") as file:
        landmark_data = json.load(file)
    
    return landmark_data

def find_max_sequence(video_ids):
    max_length = 0
    for video_id in video_ids:
        data = landmark_from_json(video_id)
        if len(data['hands_only']) > max_length: max_length = len(data['hands_only'])
        
    return max_length

def filter_on_length(video_id, length):
    
    video_data = landmark_from_json(video_id)
    
    if len(video_data['hands_only']) <= length:
        with open(f"dataset/filtered_processed_landmarks/{video_id}.json", "w") as outfile:
            json.dump(video_data, outfile)
    

if __name__ == "__main__":
    import pandas as pd
    lookup_df = pd.read_csv("dataset/wlasl_lookup.csv", converters={'video_id': str})
    print(find_max_sequence(lookup_df['video_id'].values))