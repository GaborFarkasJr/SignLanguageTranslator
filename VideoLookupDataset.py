from my_utils import *
import torch
import numpy as np
from torch.utils.data.dataloader import Dataset

class VideoLookupDataset(Dataset):
    
    def __init__(self, dataframe, encoder, sequence_length, data_aug=None):
        self.dataframe = dataframe
        self.encoder = encoder
        self.sequence_length = sequence_length
        self.data_aug = data_aug
        
    def __getitem__(self, index):
        
        row = self.dataframe.iloc[index].to_numpy()
        video_id = row[2]
        gloss = row[0]
        
        features = cv2_get_processed_video(video_id)
        label = self.encoder.encode(gloss)
        
        if self.data_aug is not None:
            # for noise in self.data_aug:
            features = self.data_aug(features)
            
        # Padding features for same sequence length batches
        if self.sequence_length is not None:
            features = pad_video(features, self.sequence_length)
        
        # Chaning the shape to [channel, frames, height, width]
        features = np.rollaxis(features, 3, 0)
        
        features, label = torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
                
        return features, label
    
    def __len__(self):
        return len(self.dataframe)