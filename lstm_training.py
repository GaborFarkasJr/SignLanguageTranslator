import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy as np
from LSTM import LSTM
from SLR_Transformer_Model import SLRTransformer
from sklearn.preprocessing import LabelEncoder
from VideoToPose import landmark_from_json
from CustomOneHot import OneHot
from my_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoLookupDataset(Dataset):
    
    def __init__(self, dataframe, encoder, input_size):
        self.dataframe = dataframe
        self.encoder = encoder
        self.input_size = input_size
        
    def __getitem__(self, index):
        
        row = self.dataframe.iloc[index].to_numpy()
        
        
        landmark_data = landmark_from_json(row[2])
        features = np.concatenate((landmark_data['lh'], landmark_data['rh'], landmark_data['p'], landmark_data['f'])) # 
        
        label = self.encoder.encode(row[0])
        
        
        # Padding features for same sequence length batches
        pad_value = self.input_size - np.size(features, -1)
        features = np.pad(features, ((0, 0), (0, pad_value)), mode='constant', constant_values=(0.0))
                    
        features, label = torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        return features, label
    
    def __len__(self):
        return len(self.dataframe)

'''
If dataset changes, only the file locations, and the lookup dataframe needs to be changed.
The splitting and training will remain the same!
'''
        
    
    
if __name__ == "__main__":
    
    lookup_df = pd.read_csv("dataset/filtered_wlasl_100_lookup.csv", converters={'video_id': str})
    
        
    
    glossary = np.unique(lookup_df['gloss'].values)
        
    # Hyperparameters
    input_size = 120 # For spotter this is the sequence of frames
    hidden_size = input_size * 2
    face_input_size = 468 # I am looking to reduce this number to only include the eyes, brows and mouth
    epoch = 400
    layers = 10
    learning_rate = 0.01
    num_classes = len(glossary)
    sequence_length =  (21 + 21 + 33 + 468) * 2 # num landmarks x dimension
    
    
    onehot_encoder = OneHot()
    onehot_encoder.fit_categories(glossary)
    
    label_encoder = LabelEncoder().fit(glossary)
    
    
    train_df = lookup_df.loc[lookup_df['split'] == 'test']    
    
    train_dataset = VideoLookupDataset(train_df, onehot_encoder, input_size)
    train_dataloader = DataLoader(train_dataset, batch_size=10)
        
    # Defining model
    
    # LSTM Model
    slr_model = LSTM(input_size, hidden_size, num_classes, layers).to(device)
    
    # slr_model = SLRTransformer(input_size, 2048, 10, num_classes, 6, 0.0)

    slr_model.to(device)
    
    
    optimiser = torch.optim.AdamW(slr_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss().to(device)

        
    train(
        num_epoch=epoch,
        dataloader=train_dataloader,
        network=slr_model,
        loss_func=loss_func,
        optimiser=optimiser,
        label_encoder=label_encoder,
        glossary=glossary
        )