import torch
from torch import nn


# ResNet34 and below 

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=None):
        super(SimpleBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # out -> out  = hidden layer essentially
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU()
        
        self.downsampling = downsampling
        
    def forward(self, x):
        
        identity = x
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        
        # Check for identity downsample
        if self.downsampling is not None:
            identity = self.downsampling(identity)
          
        return self.relu(x + identity)
    
class SimpleResNet(nn.Module):
    def __init__(self, block, video_channels, num_classes, num_layers):
        super(SimpleResNet, self).__init__()
        
        self.in_channels = 64 # Default for the model
        
        # Input Preparation
        self.conv1 = nn.Conv3d(video_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(3, 2, 1)
        
        # Blocks layers
        self.layer1 = self._make_layer(64 , block, num_layers[0], stride=1)
        self.layer2 = self._make_layer(128, block, num_layers[1], stride=2)
        self.layer3 = self._make_layer(256, block, num_layers[2], stride=2)
        self.layer4 = self._make_layer(512, block, num_layers[3], stride=2)

        # Classification
        self.ad_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_output = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, block, num_blocks, stride=1):
        downsampling = None
        layers = []
        
        # Create downsampling if needed
        if stride != 1 or self.in_channels != out_channels:
            downsampling = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, 1, stride),
                nn.BatchNorm3d(out_channels)
            )
        
        # Creating layers of residual blocks
        layers.append(block(self.in_channels, out_channels, stride, downsampling))
        self.in_channels = out_channels # Updating number of in_channels
        for _ in range(num_blocks): layers.append(block(self.in_channels, out_channels)) # Default stride no downsampling
        
        return nn.Sequential(*layers) # Each lement in layers will be treated as a block
    
    def forward(self, x):
        
        # Input
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.ad_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear_output(x)
        
        return x

# For ResNet50 and above

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=None):
        super(Block, self).__init__()
        
        self.channel_expansion = 4
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) # out -> out  = hidden layer essentially
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.channel_expansion, kernel_size=1, stride=1, padding=0) 
        self.batch_norm3 = nn.BatchNorm3d(out_channels * self.channel_expansion)
        
        self.relu = nn.ReLU()
        
        self.downsampling = downsampling
        
    def forward(self, x):
        
        identity = x
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        # Check for identity downsample
        if self.downsampling is not None:
            identity = self.downsampling(identity)
          
        return self.relu(x + identity)

class ResNet(nn.Module):
    def __init__(self, block, video_channels, num_classes, num_layers):
        super(ResNet, self).__init__()
        self.channel_expansion = 4
        
        self.in_channels = 64 # Default for the model
        
        # Input Preparation
        self.conv1 = nn.Conv3d(video_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(3, 2, 1)
        
        # Blocks layers
        self.layer1 = self._make_layer(64 , block, num_layers[0], stride=1)
        self.layer2 = self._make_layer(128, block, num_layers[1], stride=2)
        self.layer3 = self._make_layer(256, block, num_layers[2], stride=2)
        self.layer4 = self._make_layer(512, block, num_layers[3], stride=2)
        
        # Classification
        self.ad_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_output = nn.Linear(512 * self.channel_expansion, num_classes)
        
    def _make_layer(self, out_channels, block, num_blocks, stride=1):
        downsampling = None
        layers = []
        
        # Create downsampling if needed
        if stride != 1 or self.in_channels != out_channels * self.channel_expansion:
            downsampling = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * self.channel_expansion, 1, stride),
                nn.BatchNorm3d(out_channels * self.channel_expansion)
            )
        
        # Creating layers of residual blocks
        layers.append(block(self.in_channels, out_channels, stride, downsampling))
        self.in_channels = out_channels * self.channel_expansion # Updating number of in_channels
        for _ in range(num_blocks): layers.append(block(self.in_channels, out_channels)) # Default stride no downsampling
        
        return nn.Sequential(*layers) # Each lement in layers will be treated as a block
    
    def forward(self, x):
        
        # Input
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.ad_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear_output(x)
        
        return x

# The most important is the num_layers -> this defines the ResNet50
def resnet50_classifier(video_channels, num_classes):
    return ResNet(Block, video_channels, num_classes, [3, 4, 6, 3])

def resnet34_classifier(video_channels, num_classes):
    return SimpleResNet(SimpleBlock, video_channels, num_classes, [3, 4, 6, 3])

def resnet18_classifier(video_channels, num_classes):
    return SimpleResNet(SimpleBlock, video_channels, num_classes, [2, 2, 2, 2])

class ResnetMlpClassifier(nn.Module):
    def __init__(self, num_classes, classifier_hidden_size):
        super(ResnetMlpClassifier, self).__init__()
    
        self.resnet = resnet34_classifier(3, classifier_hidden_size)
        
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(classifier_hidden_size, num_classes))
        
    def forward(self, x):
        
        x = self.resnet(x)
        return self.mlp(x)


        