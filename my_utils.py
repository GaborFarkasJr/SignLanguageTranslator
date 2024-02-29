import torch
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(dataloader, model, loss_func, optimiser):
    
    model, loss_func = model.to(device).train(), loss_func.to(device)
    torch.enable_grad()
    
    correct = 0
    total = 0
    running_loss = 0
    
    start_time = time()
    for features, labels in dataloader:
        
        labels = labels.transpose(0, 1)
        
        # Moving to gpu
        features = features.to(device)
        labels = labels.to(device)
        
        optimiser.zero_grad()
        logits = model(features)
        
        loss = loss_func(logits[0], labels[0])
        loss.backward()
        optimiser.step()
        
        predicted = logits[0].argmax(-1)
        real = labels[0].argmax(-1)
        
        # Accuracy and Running Loss
        running_loss += loss
        correct += (predicted == real).sum().item()
        total += labels.size(0)
    
    duration = time() - start_time
    total_loss = running_loss / total
    total_accuracy = correct / total
    
    torch.cuda.empty_cache()
    return total_loss, total_accuracy, duration
        
    

# Can be used for testing as well
def validate_one_epoch(dataloader, model, loss_func):
    
    model, loss_func = model.to(device).eval(), loss_func.to(device)
    torch.no_grad()

    correct = 0
    total = 0
    running_loss = 0
    
    start_time = time()
    for features, labels in dataloader:
        
        labels = labels.transpose(0, 1)
        
        # Moving to gpu
        features = features.to(device)
        labels = labels.to(device)
        
        logits = model(features)
        
        loss = loss_func(logits[0], labels[0])
        
        predicted = logits[0].argmax(-1)
        real = labels[0].argmax(-1)
        
        # Accuracy and Running Loss
        running_loss += loss
        correct += (predicted == real).sum().item()
        total += labels.size(0)
         
    duration = time() - start_time
    total_loss = running_loss / total
    total_accuracy = correct / total
    
    torch.cuda.empty_cache()
    return total_loss, total_accuracy, duration
    
    
