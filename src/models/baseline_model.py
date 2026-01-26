import torch #pytorch library
import torch.nn as nn #pytorch neural network module

class BaselineModel(nn.Module): #baseline model class, inherits from base pytorch class
    def __init__(self, hidden_channels: int = 64): #initialize the model with 64 internal feature maps (can be changed)

        super().__init__() #call parent class initializer

        self.net = nn.Sequential( #sequential container to hold layers, runs in order
            #using convolutional layers to learn local patterns 
            nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, padding=1), #input is 1 channel (grayscale sketch), output is hidden_channels (64) feature maps (begin learning patterns
            nn.ReLU(), #activation function to introduce non-linearity
            #with kernel size 3, model looks at 3x3 so 9 pixels at a time, padding = 1 ensures the kernel can slide over edges, without shrinking image size per pass

            #second conv layer
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1), #input is hidden_channels (64) feature maps, output is same number of feature maps
            nn.ReLU(), #activation function

            #third conv layer
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1), #input is hidden_channels (64) feature maps, output is same number of feature maps
            nn.ReLU(), #activation function

            #fourth conv layer
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1), #input is hidden_channels (64) feature maps, output is same number of feature maps
            nn.ReLU(), #activation function

            # fifth conv layer
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1), #input is hidden_channels (64) feature maps, output is same number of feature maps

            #batchNorm layer to stabilize training
            nn.BatchNorm2d(hidden_channels),
            
            nn.ReLU(), #activation function

            # final conv layer maps to 3 channels (RGB color image)
            nn.Conv2d(in_channels=hidden_channels, out_channels=3, kernel_size=1), 
            # 1 kernel size ensures there isnt any neighbourhood lookup, the 3 output channels represent RGB

            # squash output between range -1 -> 1 to match dataset normalization
            #nn.Tanh() #removed due to outputs being fully blacked out images

        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: #forward pass method, takes in a tensor (grayscale version) and returns a tensor (predicted color version)

        return self.net(x) #pass input x through the sequential network defined above and return the output
    # x represents a batch of sketches, output will be corresponding batch of predicted color images




        
