from video_classifier import *
from my_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_checkpoints/Resnet_3333_30fps_122res/9e_Resnet_3333_30fps_122res.pt"


class model_instance():
    
    
    def __init__(self, buffer_size = 50, min_confidence = 0.5):
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        
        self.model = SimpleResNet(SimpleBlock, 3, 63, [3, 3, 3, 3]).to(device)
        self.glossary = load_model(model_path, self.model)['gloss_list']
        
    def set_buffer_size(self, n): 
        self.buffer_size = n
    
    def set_min_confidence(self, n):
        self.min_confidence = n/100
        
    def __call__(self, video):
        
        video = torch.from_numpy(video).to(device)
        video = video.unsqueeze(0)
        video = video.type(torch.float32)
        prediction = self.model(video)
        prediction = torch.softmax(prediction, 1)
        gloss_index = torch.argmax(prediction, 1)
        
        if prediction[0][gloss_index] >= self.min_confidence:
            return self.glossary[gloss_index], prediction[0][gloss_index].item()
        else: 
            return None