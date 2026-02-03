import torch # Importing the main PyTorch library
import torch.nn as nn # Importing the PyTorch neural network module

class Discriminator(nn.Module):
    def __init__(self, in_channels = 4): #combines L (1 channel) and RGB (3 channels) = 4 channels
        super().__init__() #initialize parent nn.Module class

        def conv_block(in_channels, out_channels, stride): #takes in in_channels, returns out_channels, takes in stride
            return nn.Sequential( #sequential container to stack layers
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1), #4x4 window over pixels, 1 stride ensures images arent shrunk too fast, padding 1 to maintain spatial dimensions
                nn.BatchNorm2d(out_channels), # normalizes activations to improve training stability
                nn.LeakyReLU(0.2, inplace=True) #similar to ReLu but allows small gradient when negative
            )
        self.model = nn.Sequential( #main model sequential container
            conv_block(in_channels, 64, stride=2), #first conv block, takes in 4 channels, outputs 64 channels, stride 2 to downsample (64 channels learns simple patterns)
            conv_block(64, 128, stride=2), #second conv block, takes in 64 channels, outputs 128 channels, stride 2 to downsample (128 channels learns more complex patterns)
            conv_block(128, 256, stride=2), #third conv block, takes in 128 channels, outputs 256 channels, stride 2 to downsample (256 channels learns even more complex patterns and global contexts)
            conv_block(256, 512, stride=1), #fourth conv block, takes in 256 channels, outputs 512 channels, stride 1 to maintain size (deep part of network, learns high-level features)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1), #final conv layer to output single channel (real/fake score)
            # output shape (B , 1 ,30 , 30), gives a 'realness score' for each 4x4 patch in the image
        )
    def forward(self, x): #forward pass method
        return self.model(x) #pass input x through the model and return the output