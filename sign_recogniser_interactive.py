import cv2
import numpy as np
from pose_estimation.scripts.pose_api import PoseApi
from model_instance import model_instance
import torch
from my_utils import *
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

        

api = PoseApi()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_blank_frame(height, width, r, g, b):
    blank_frame = np.zeros((height, width, 3), np.uint8)
    blank_frame[:, :, 0] = b
    blank_frame[:, :, 1] = g
    blank_frame[:, :, 2] = r
    
    return blank_frame

if __name__ == "__main__":
    
    display_size = 500
    
    model = model_instance()
    cap = cv2.VideoCapture(0)
    height, width = 122, 122
    b, g, r = 0x00, 0x00, 0x00  # black
    
    '''
    0 -> normal
    1 -> skeleton
    '''
    display_type = 0
    
    
    
    processing_buffer = []
    predicted_output = "Word: none predicted"
    can_process = False
    frame_count = 0
    
    # Display frame
    display_frame = None
    cv2.namedWindow("Sign Interactive")
    cv2.createTrackbar("buffer_slider", "Sign Interactive", 100, 100, model.set_buffer_size)
    cv2.setTrackbarPos("buffer_slider", "Sign Interactive", 50)
    cv2.createTrackbar("confidence_slider", "Sign Interactive", 100, 100, model.set_buffer_size)
    cv2.setTrackbarPos("confidence_slider", "Sign Interactive", 50)
    
    
    
    ret, cap_frame = cap.read()
    while cap.isOpened():
        
        if ret == True:
            frame_count += 1
            # v to switch between views
            if cv2.waitKey(10) & 0xFF == ord('v'):
                display_type = (display_type + 1) % 2
                    
            processing_frame = create_blank_frame(display_size, display_size, r, g, b)
            
                
            # Converting frame image from BGR to RGB
            cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)    
            
            display_frame = cv2.resize(cap_frame, (display_size, display_size))
                
            # Feature estimation
            # face_results = api.face_landmarks_from_image(cap_frame)
            hand_results = api.hand_landmark_from_image(cap_frame)
            pose_results = api.pose_landmarks_from_image(cap_frame)
            
            # Drawing features
            # api.draw_face_landmarks(processing_frame, face_results)
            api.draw_hand_landmarks(processing_frame, hand_results)
            api.draw_pose_landmarks(processing_frame, pose_results)
        
            # Prediction
            if len(processing_buffer) >= model.buffer_size:
                
                processing_buffer = np.stack(processing_buffer, axis=1)
                predicted_word = model(processing_buffer)
                processing_buffer = []
                
                # show whenever buffer is cleared
                cv2.putText(display_frame, "*", (display_size - 25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                
                # Printing on screen and also on console for debugging
                if predicted_word is not None: 
                    predicted_output = f"Word: {predicted_word[0]}"
                    sys.stdout.flush()
                    sys.stdout.write(f"\rWord: {predicted_word[0]} Confidence: {predicted_word[1]}                          ")
                else:
                    predicted_output = "Word: none predicted"
                    sys.stdout.flush()
                    sys.stdout.write(f"\rWord: none predicted                          ")
            processing_buffer.append(np.rollaxis(cv2.resize(processing_frame, (height, width)), -1, 0)); 
            
            
            
            cv2.putText(display_frame, predicted_output, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.imshow("Sign Interactive", np.concatenate([display_frame, processing_frame], axis=1))
           
        else:
            break
        # q to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        ret, cap_frame = cap.read()