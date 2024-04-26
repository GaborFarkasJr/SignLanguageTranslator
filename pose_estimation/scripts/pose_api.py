import mediapipe as mp

class PoseApi:
    
    def __init__(self) -> None:
        pass
    
    # Setting up with default values    
    
    mp_hands = mp.solutions.hands.Hands(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.3,
        model_complexity = 1,
    )
    mp_face = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
    )
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
        model_complexity = 1,
    )
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
   
    # ----> Processing Results
    
    # Process and find information about the detected hands
    def hand_landmark_from_image(self, rgb_image):
        return self.mp_hands.process(rgb_image)
    
    # Process and find information about the detected face
    def face_landmarks_from_image(self, rgb_image):
        return self.mp_face.process(rgb_image)
    
    # Process and find information about the detected pose
    def pose_landmarks_from_image(self, rgb_image):
        return self.mp_pose.process(rgb_image)
    
    # ----> Drawing landmarks
    
    # Draw the hand landmarks onto a provided image
    def draw_hand_landmarks(self, rgb_image, process_results):
        if process_results.multi_hand_landmarks:
            for hand_landmarks in process_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image = rgb_image,
                    landmark_list = hand_landmarks,
                    connections = mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )
    # Draw the face_mesh and landmarks into a provided image
    def draw_face_landmarks(self, rgb_image, process_results):
        if process_results.multi_face_landmarks:
            for face_landmarks in process_results.multi_face_landmarks:
                # self.mp_drawing.draw_landmarks(
                #     image = rgb_image,
                #     landmark_list = face_landmarks,
                #     connections = mp.solutions.face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec = None,
                #     connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                # )
                self.mp_drawing.draw_landmarks(
                    image = rgb_image,
                    landmark_list = face_landmarks,
                    connections = mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    # connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                 )
                # self.mp_drawing.draw_landmarks(
                #     image = rgb_image,
                #     landmark_list = face_landmarks,
                #     connections = mp.solutions.face_mesh.FACEMESH_LIPS,
                #     landmark_drawing_spec = None,
                #     connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_contours_style()
                # )
                
    def draw_pose_landmarks(self, rgb_image, process_results):
        if process_results.pose_world_landmarks:
            self.mp_drawing.draw_landmarks(
                image = rgb_image,
                landmark_list = process_results.pose_landmarks,
                connections = mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec = None,
                connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )
                
    # ----> Flattebed Landmarks

    # 2D
    def get_hand_landmark_coordinates(self, process_results):
        
        ''' 
                    Currently, with mediapipe, there is a chance that the right hand will be detected as left hand for individual frames. The left hand will
                    then have 2 or 3 times the number of landmarks (42 or 63). This most likely happens when the two hands are overlapping and the image is 
                    not clear enough. The current fix for this problem is to just return None for both the hands. There is no way to tell which order the 
                    landmarks come in if multiple are detected in one hand.
        '''
        
        
        landmarks_dictionary = {
            'Left': [0.0 for i in range(21 * 2)],
            'Right': [0.0 for i in range(21 * 2)]
        }
        
        if process_results.multi_hand_landmarks:
            
            for hand_index in range(len(process_results.multi_hand_landmarks)):
                hand_label = process_results.multi_handedness[hand_index].classification[0].label
                        
                index = 0
                for landmark_id, landmark in enumerate(process_results.multi_hand_landmarks[hand_index].landmark):    
                    landmarks_dictionary[hand_label][index] = round(landmark.x, 4)
                    landmarks_dictionary[hand_label][index + 1] = round(landmark.y, 4)
                    
                    index += 2
                    
                    # Sometimes two hands are detected
                    if landmark_id >= 21:
                        break
        
        return landmarks_dictionary['Left'], landmarks_dictionary['Right']
    
    # 3D
    def get_3d_hand_landmark_coordinates(self, process_results):
        
        ''' 
                    Currently, with mediapipe, there is a chance that the right hand will be detected as left hand for individual frames. The left hand will
                    then have 2 or 3 times the number of landmarks (42 or 63). This most likely happens when the two hands are overlapping and the image is 
                    not clear enough. The current fix for this problem is to just return None for both the hands. There is no way to tell which order the 
                    landmarks come in if multiple are detected in one hand.
        '''
        
        
        landmarks_dictionary = {
            'Left': [0.0 for i in range(21 * 3)],
            'Right': [0.0 for i in range(21 * 3)]
        }
        
        if process_results.multi_hand_landmarks:
            
            for hand_index in range(len(process_results.multi_hand_landmarks)):
                hand_label = process_results.multi_handedness[hand_index].classification[0].label
                        
                index = 0
                for landmark_id, landmark in enumerate(process_results.multi_hand_landmarks[hand_index].landmark):    
                    landmarks_dictionary[hand_label][index] = round(landmark.x, 4)
                    landmarks_dictionary[hand_label][index + 1] = round(landmark.y, 4)
                    landmarks_dictionary[hand_label][index + 2] = round(landmark.z, 4)
                    
                    index += 3
                    
                    # Sometimes two hands are detected
                    if landmark_id >= 21:
                        break
        
        return landmarks_dictionary['Left'], landmarks_dictionary['Right']
    
    # 2D
    def get_face_landmark_coordinates(self, process_results):
        face_landmarks = [0.0 for _ in range(478 * 2)]
        if process_results.multi_face_landmarks:
            
            # multi_face_landmarks is a list of faces, we use only 1 face so it is set to 0
            
            index = 0
            for landmark_id, landmark in enumerate(process_results.multi_face_landmarks[0].landmark):
                face_landmarks[index] = round(landmark.x, 4)
                face_landmarks[index + 1] = round(landmark.y, 4)
                
                index += 2
            
                # Incase two faces are detected
                if landmark_id >= 478:
                    break
                    
        return face_landmarks
    
    # 3D
    def get_3d_face_landmark_coordinates(self, process_results):
        face_landmarks = [0.0 for _ in range(478 * 3)]
        if process_results.multi_face_landmarks:
            
            # multi_face_landmarks is a list of faces, we use only 1 face so it is set to 0
            
            index = 0
            for landmark_id, landmark in enumerate(process_results.multi_face_landmarks[0].landmark):
                face_landmarks[index] = round(landmark.x, 4)
                face_landmarks[index + 1] = round(landmark.y, 4)
                face_landmarks[index + 2] = round(landmark.z, 4)
                
                index += 3
            
                # Incase two faces are detected
                if landmark_id >= 478:
                    break
                    
        return face_landmarks

    # 2D
    def get_pose_landmark_coordinates(self, process_results):
        pose_landmarks = [0.0 for _ in range(33 * 2)]
        if process_results.pose_landmarks:
            
            # print(process_results.pose_landmarks.landmark)
            
            index = 0
            for landmark_id, landmark in enumerate(process_results.pose_landmarks.landmark):
                pose_landmarks[index] = round(landmark.x, 4)
                pose_landmarks[index + 1] = round(landmark.y, 4)
                
                index += 2
            
                # Incase two faces are detected
                if landmark_id >= 66:
                    break
                    
        return pose_landmarks
    
    # ----> Landmarks from index

    def hand_landmark_from_index(self, process_results, index):
        
        ''' 
        Designed to be veratile with different implementations. There are 21 landmarks (0 to 21).
        If there are no landmarks found, the model will return 0.
        '''
        landmarks_dictionary = {
            'Left': {
                'x': 0.0,
                'y': 0.0
            },
            'Right': {
                'x': 0.0,
                'y': 0.0
            }
        }

        if process_results.multi_hand_landmarks:
            
            for hand_index in range(len(process_results.multi_hand_landmarks)):
                hand_label = process_results.multi_handedness[hand_index].classification[0].label
                        
                landmarks = process_results.multi_hand_landmarks[hand_index].landmark
                
                landmarks_dictionary[hand_label]['x'] = landmarks[index].x
                landmarks_dictionary[hand_label]['y'] = landmarks[index].y
                    
        
        return landmarks_dictionary['Left'], landmarks_dictionary['Right']
    
    def face_landmark_from_index(self, process_results, index):
        face_landmarks = {
                'x': 0.0,
                'y': 0.0
            }
        if process_results.multi_face_landmarks:
                        
            landmarks = process_results.multi_face_landmarks[0].landmark
            
            
            face_landmarks['x'] = landmarks[index].x
            face_landmarks['y'] = landmarks[index].y
            
                    
        return face_landmarks
    
    def pose_landmark_from_index(self, process_results, index):
        pose_landmarks = {
                'x': 0.0,
                'y': 0.0
            }
        if process_results.pose_landmarks:
            
            # print(process_results.pose_landmarks.landmark)
            
            landmarks = process_results.pose_landmarks.landmark
            
            pose_landmarks['y'] = landmarks[index].y
            pose_landmarks['x'] = landmarks[index].x
            
                    
        return pose_landmarks
    
    
    