import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data.dataloader import DataLoader
from video_classifier import *
from CustomOneHot import OneHot
from my_utils import *
from VideoLookupDataset import VideoLookupDataset
# Imported some other resnet models as well, for testing

# Argparse
parser = argparse.ArgumentParser(description=
                                 "This is the script that is used to train the sign language recognition model. Before running this script it is important "
                                 "that you have some of the following:\n"
                                 "* The videos for training within 'dataset/processed/'\n"
                                 "* The .csv lookup file for the processed videos within 'dataset/'\n"
                                 "For more instructions go to the'README.md' file",
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("-mn", "--model_name", metavar="<name>", default="SLR_Classifier", help="Will create the .pt model files and the logs under this name", type=str)
parser.add_argument("-cf", "--checkpoint_frequency", metavar="<frequency>", default=10 , help="Saves model as a .pt file after every <frequency> epoch. " 
                    "epoch (index from training loop) is also attached to the file name", type=int)
parser.add_argument("-e", "--epochs", metavar="<epochs>", default=20, help="Number of epochs (i.e repetitions) training will run through", type=int)
parser.add_argument("-lr", "--learning_rate", metavar="<learning_rate>", default=0.001, help="The learning rate used when updating model parameters", type=float)
parser.add_argument("-sl", "--sequence_length", metavar="<sequence_length>", default=None, help="Training data will be padded to this length. "
                    "By default, there is no padding.", type=int)
parser.add_argument("-mc", "--model_checkpoint", metavar="<model_checkpoint>", default=None, help="If specified, checkpoint will be loaded and carried on", type=str)
parser.add_argument("-l", "--lookup", metavar="<lookup>", default="dataset/LSA64_lookup.csv", help="the path to the .csv lookup file containing the data for training", type=str)

args = parser.parse_args()

# Model essentials
model_name = args.model_name
model_checkpoint = args.model_checkpoint
lookup_path = args.lookup

# Hyperparameters
num_epochs = args.epochs
checkpoint_frequency = args.checkpoint_frequency
learning_rate = args.learning_rate
sequence_length = args.sequence_length

# Folders for saving checkpoints and training logs
if not os.path.isdir(f"training_logs/{model_name}/"):
    os.mkdir(f"training_logs/{model_name}/")
    print(f"training_logs folder for {model_name} has been made")
if not os.path.isdir(f"model_checkpoints/{model_name}/"):
    os.mkdir(f"model_checkpoints/{model_name}/")
    print(f"model_checkpoints folder for {model_name} has been made")

if __name__ == "__main__":
    
    lookup_df = pd.read_csv(lookup_path, converters={'video_id': str})
    glossary = np.unique(lookup_df['gloss'].values)
    
    # glossary = glossary[:25] # Training for 25
    num_classes = len(glossary)
    
    # Onehot encoder
    onehot_encoder = OneHot()
    onehot_encoder.fit_categories(glossary)
    
    
    # Train data
    train_df = lookup_df.loc[lookup_df['split'].isin(['train', 'val'])]
    train_df = train_df.loc[train_df['gloss'].isin(glossary)]      
    train_dataset = VideoLookupDataset(train_df, onehot_encoder, None) # no need to pad if batch size is 1
    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=4) # Computer cant handle more videos at once :D
    
    # Val data
    eval_df = lookup_df.loc[lookup_df['split'] == 'test']
    eval_df = eval_df.loc[eval_df['gloss'].isin(glossary)] 
    eval_dataset = VideoLookupDataset(eval_df, onehot_encoder, None)
    eval_dataloader = DataLoader(eval_dataset, pin_memory=True, num_workers=4)
        
    # Defining model
    # Additional resnet 18 with increased blocks
    # slr_model = SimpleResNet(SimpleBlock, 3, num_classes, [3, 3, 3, 3]).to(device)
    slr_model = resnet18_classifier(3, num_classes).to(device)

    optimiser = torch.optim.SGD(slr_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    # Training and evaluation
    progress_data = pd.DataFrame(columns=["epoch", "train_loss", "eval_loss", "train_acc", "eval_acc"])
    
    # Loading checkpoint
    from_epoch = 0
    if model_checkpoint is not None:
        model_data = load_model(model_checkpoint, slr_model)
        optimiser, from_epoch = model_data['optimiser'], model_data['epoch']
        from_epoch += 1
        
    # Training loop
    for epoch in range(from_epoch, num_epochs):
        
        # Train and evaluation with metric return
        t_l, t_a = train(slr_model, train_dataloader, loss_func, optimiser)
        torch.cuda.empty_cache()
        e_l, e_a = evaluate(slr_model, eval_dataloader, loss_func)
        torch.cuda.empty_cache()
        
        # Progress update
        progress_data = update_training_progress(progress_data, epoch+1, t_l, e_l, t_a, e_a)

        # Saving checkpoint and creating log
        if (epoch + 1) % checkpoint_frequency == 0:
            progress_data.to_csv(f"training_logs/{model_name}/log.csv", header=True, index_label=False)
            save_model(f'model_checkpoints/{model_name}/{epoch}e_{model_name}', slr_model, optimiser, epoch)
        
        # Printing info
        print("\n")
        print(progress_data)
        
    print("Training Finished")
      
        
    
    