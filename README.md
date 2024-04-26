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
The project is already set up in a way that allows the model to be run and used once the files are installed and placed in the correct locations as indicated above. However, If you wish to create the model from scratch or make changes you can do that as well.
### pre-processing
1. Download the LSA64 videos and place them in `dataset/videos`.
2. Download the `LSA64_videos_test.csv` and `LSA64_videos_train.csv` files and place them in `dataset/`.
3. To set up the lookup table, run `create_lookup.py`.
4. To create the preprocessed videos, run `classifier_preprocessing.py`. This step might take quite a few hours, so it will be available to download. The preprocessed videos are 244 x 244 and 60 fps (although rendered video might be 25fps).
### Training
**It is important to note that during training, the videos are resized to 144 x 144 and every other frame is taken to mimmic a 30fps video. If a different dataset is used, make sure `my_utils.py` is checked out**.
1. Make sure that the file structure matches the one stated above. If changes were made, make sure that the files are accessed properly.
2. Run `test.py` to train the model. If you do `python test.py --help`, there will be several options for hyperparameters. alternatively, default values are already set up and can just be run on its own.
### Post-processing
1. Make a note of where the saved model is stored and the lookup csv file.
2. Run `add_glossary.py` to add the glossary. For modre details, use `python add_glossary.py --help`.
3. After adding the glossary terms, rearrange the files so that it matches the above.
4. Alternatively, you can change the **model_path** variable within `model_instance.py` to point towards the .pt model file with the glossary terms added.
### running the application
1. Make sure a web camera is connected, since the interactive applicaiton is only designed for it.
2. Check to see if the file structure matches the above. If you have modified `model_instance.py`, make sure that the file points to the correct direction.
3. Run `sign_recogniser_interactive.py`.
4. Play around with the buffer size and the min-confidence sliders for different effects.
## Attributions
- The LSA64 dataset which can be found <a href="https://facundoq.github.io/datasets/lsa64/">here</a>.
```
@Article{Ronchetti2016,
author="Ronchetti, Franco and Quiroga, Facundo and Estrebou, Cesar and Lanzarini, Laura and Rosete, Alejandro",
title="LSA64: A Dataset of Argentinian Sign Language",
journal="XX II Congreso Argentino de Ciencias de la Computación (CACIC)",
year="2016"
}
```
- The two seprated lookup csv files for the training and testing were from Mark Marais's <a href="https://www.kaggle.com/code/marcmarais/lsa64-signer-independence-inceptionv3-rnn/input">Kaggle Project</a>
- Mediapie's <a href="https://github.com/google/mediapipe">Github page</a> and <a href="https://developers.google.com/mediapipe">webpage</a>
