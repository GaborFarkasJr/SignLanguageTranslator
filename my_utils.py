import torch
import numpy as np
import cv2
import skvideo.io as skio
import sys
from time import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(model, train_data, loss_func, optimiser):
    correct = 0
    total = 0
    running_loss = 0
    
    model.train()
    running_duration = 0
    for batch, (features, labels) in enumerate(train_data):
        labels = labels.transpose(0, 1)[0]
        start_time = time()

        # Moving to gpu
        features = features.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        logits = model(features)

        loss = loss_func(logits, labels)
        
        predicted = logits[0].argmax(-1)
        real = labels.argmax(-1)
        
        # Accuracy and Running Loss
        running_loss += loss.item()
        correct += (predicted == real).sum().item()
        total += labels.size(0)
        
        
        loss.backward()
        optimiser.step()
        
        running_duration += (time() - start_time)
        
        sys.stdout.flush()
        sys.stdout.write(f"\r{batch}/{len(train_data)} loss: {running_loss / (batch + 1)} {((len(train_data) - batch) * (running_duration / (batch+1))) / 60} minutes")
        
        
        torch.cuda.empty_cache()
            
    train_total_loss = running_loss / len(train_data)
    train_total_accuracy = correct / total
        
    
    return train_total_loss, train_total_accuracy

def evaluate(model, eval_data, loss_func):
    
    

    correct = 0
    total = 0
    running_loss = 0
    
    model.eval()
        
    with torch.no_grad():
        
        
        running_duration = 0
        for batch, (features, labels) in enumerate(eval_data):    
            start_time = time()
            labels = labels.transpose(0, 1)[0]
            
            # Moving to gpu
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            
            loss = loss_func(logits, labels)
            
            predicted = logits.argmax(-1)
            real = labels.argmax(-1)
            
            # Accuracy and Running Loss
            running_loss += loss.sum().item()
            correct += (predicted == real).sum().item()
            total += labels.size(0)
            
            running_duration += (time() - start_time)
            
            sys.stdout.flush()
            sys.stdout.write(f"\r{batch}/{len(eval_data)} loss: {running_loss / (batch + 1)} {((len(eval_data) - batch) * (running_duration / (batch+1))) / 60} minutes")
            
            
            torch.cuda.empty_cache()
            
    valid_total_loss = running_loss / len(eval_data)
    valid_total_accuracy = correct / total
    
    return valid_total_loss, valid_total_accuracy


# plot and plot_data are arrays with the two indexes matching
def update_training_progress(df, epoch, train_loss, eval_loss, train_acc, eval_acc):    
    df.loc[len(df.index)] = [epoch, train_loss, eval_loss, train_acc, eval_acc]
    return df
    
def pad_video(video, max_frames):
    pad_value = max_frames - np.size(video, 0)
    if pad_value != 0:
        pad_frame = np.zeros(np.shape(video[0]), dtype=np.int32)
        return np.append(video, [pad_frame ] * pad_value, 0)
    else: return video

def get_processed_video(video_id):
    return skio.vread(f"dataset/processed/{video_id}.mp4")

def cv2_get_processed_video(video_id):
    video = []

    cap = cv2.VideoCapture(f"dataset/processed/{video_id}.mp4")
    ret = True
    frame_count = 0
    while ret:
        ret, frame = cap.read()
        
        if ret: 
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_count % 2 == 0: # reading video as 30 fps, since video is 60
                video.append(cv2.resize(frame, (122, 122)))
        frame_count += 1
    return np.stack(video, axis=0)

def cv2_original_video(video_id):
    video = []

    cap = cv2.VideoCapture(f"dataset/videos/{video_id}.mp4")
    ret = True
    while ret:
        ret, frame = cap.read()
        
        if ret: 
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.append(cv2.resize(frame, (244, 244)))
    return np.stack(video, axis=0)
    

def save_model(file_path, model, optimiser=None, epoch=None):
    '''Saves and adds ".pt". Optimiser and epoch only saved when needed'''
    
    save_data = {
        'model_state_dict': model.state_dict()
    }
    if optimiser is not None: save_data['optimiser'] = optimiser
    if epoch is not None: save_data['epoch'] = epoch
    
    torch.save(save_data, f'{file_path}.pt')
    
def load_model(file_path, model):
    '''
    Returns the model data loaded from torch.load(). It also automatically loads weights
    '''
    model_data = torch.load(file_path)
    model.load_state_dict(model_data['model_state_dict']) # Loading weights to model
    
    return model_data

# In case classifier labels were not added:
def add_model_labels(file_path, gloss_list):
    model_data = torch.load(file_path)
    model_data['gloss_list'] = gloss_list
    torch.save(model_data, file_path)
    
def model_predict(model, video, glossary, confidence):
    # video = np.rollaxis(video, 4, 1)
    video = torch.from_numpy(video).to(device)
    video = video.unsqueeze(0)
    video = video.type(torch.float32)
    prediction = model(video)
    prediction = torch.softmax(prediction, 1)
    gloss_index = torch.argmax(prediction, 1)
    
    if prediction[0][gloss_index] >= confidence:
        return glossary[gloss_index], prediction[0][gloss_index].item()
    else: 
        return None