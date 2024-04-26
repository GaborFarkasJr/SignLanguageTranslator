# Pose Assisted Method for Isolated Sign Language Classification
This is a sign language recognition (more specifically classification) project that focuses on finding a flexible and intuitive method to classify isolated signs. The methodology and the set-up is designed to be easy-to-understand. For this, there is a detailed section to use and set up the files. This project is designed for the LSA64 dataset and mediapipe, but the scripts can be re-written.
## Files and uses
- `dataset/LSA64_lookup.csv` - Used to map the glossary terms of the LSA64 dataset to the videos.
- `dataset/create_lookup.py` - Creates the lookup csv files for training and testing.
- `pose_estimation/scripts/api_test.py` - Just a script to play around with and test `pose_api.py`.
- `pose_estimation/scripts/pose_api.py` - Pose-estimation script using Mediapipe. Contains code for facial estimation but it isn't used in final model.
- `add_glossary.py` - Adds the list of glossary terms onto the model file. The final model already has the terms added.
- `classifier_preprocessing.py` - Preprocessed the LSA64 dataset videos using `draw_skeleton.py`.
- `CustomOneHot.py` - Adds one-hot encoding to the labels during training and testing.
- `draw_skeleton.py` - Converts LSA64 videos to a "skeleton" format by connecting the predicted landmarks.
- `model_instance.py` - Creates a model instance with built in functions to use for predictions.
- `my_utils.py` - Important functions that is used by several other scripts.
- `sign_recogniser_interactive.py` - Interactive application to test the model with webcam.
- `train.py` - Trains the model and saves model periodically.
- `video_log_analysis.ipynb` - Draws the accuracy of the model accross several epochs from a csv log file.
- `video_classifier.py` - Contains the 3D ResNet implementation used for this project.
- `VideoLookupDataset.py` - Fetches the Preprocessed videos and adds labels for training and testing.
## File Structure
## Files to Install
## Setup
## Attributions
