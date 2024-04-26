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
## Prerequisites
If you are using a GPU to run the application, check the version of CUDA and CUDNN that is compatible with your computer. Mediapipe is made with Tensorflow and therefore cannot run on the GPU. However, this model is made with Pytorch, and can run on a GPU.
- Python 3.11.5 (other versions might work but check compatability with the others)
- Mediapipe
- OpenCV
- Numpy
- Pytorch (torch)
- argparse
- pandas
- matplotlib (for plotting the graph in `training_log_analysis.ipynb`)
- skvideo
- multiprocessing
- sklearn
- CUDA (Only if a GPU is available)
- CUDNN (Only if a GPU is available)
## File Structure
```
│   add_glossary.py
│   classifier_preprocessing.py
│   CustomOneHot.py
│   draw_skeleton.py
│   model_instance.py
│   my_utils.py
│   README.md
│   sign_recogniser_interactive.py
│   train.py
│   training_log_analysis.ipynb
│   VideoLookupDataset.py
│   video_classifier.py
│
├───dataset
|   |   LSA64_videos_test.csv
|   |   LSA64_videos_train.csv
│   │   create_lookup.py
│   │   LSA64_lookup.csv
│   │
│   ├───processed
│   │       001_001_001.mp4
|   |       ...
│   │
│   └───videos
│   │       001_001_001.mp4
|   |       ...
|   |
├───final_model
│   └───Resnet_3333_30fps_122res
│           9e_Resnet_3333_30fps_122res.pt
│           log.csv
│
└───pose_estimation
    └───scripts
            api_test.py
            pose_api.py
```
`dataset/videos` only needed when pre-processing. Otherwise, the training will only the `dataset/processed`, which can be downloaded in the next section. For the creation of the lookup csv, `LSA64_videos_test.csv` and `LSA64_videos_train.csv` are needed and can be downloaded in next section, but is also provided.
## Files to Install
- Final model via <a href="https://drive.google.com/file/d/1-JikMiyT7U7OwS8DmUZHkEaJJWLxiw9D/view?usp=sharing">Google Drive</a>
- Pre-processed videos via <a href="https://drive.google.com/file/d/16lb2ysbMgvH3-vl5vO4K_HMl3KNyF8xK/view?usp=sharing">Google Drive</a>
- Original Videos used for training via <a href="https://facundoq.github.io/datasets/lsa64/">LSA64 website</a>
- `LSA64_videos_test.csv` and `LSA64_videos_train.csv` CSV files used in `dataset/create_lookup.py` from <a href="https://www.kaggle.com/code/marcmarais/lsa64-signer-independence-inceptionv3-rnn/input">Kaggle</a>

## Setup
## Attributions
