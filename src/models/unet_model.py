#UNET MODEL

import torch # Import PyTorch library
import torch.nn as nn # Import neural network module
import torch.nn.functional as F # Import functional module for activation functions

class DoubleConv(nn.Module):
    # Double convolution block, 2 convolutional layers to extract features and combinations
    def __init__(self, in_channels, out_channels): #initialize with input channels (1 from the sketch) and output channels 
        super().__init__() #register parameters for pytorch

        # Define the double convolutional layers
        self.block = nn.Sequential( # Sequential container to hold layers, runs in order
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # First convolutional layer, kernel size 3 so a3x3 window is applied, learns local detail and patterns
            #padding 1 to maintain spatial dimensions
            nn.BatchNorm2d(out_channels), # Batch normalization for stability, normalise feature values
            nn.ReLU(inplace=True), # ReLU activation function, introduce non-linearity
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # Second convolutional layer
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.ReLU(inplace=True) # ReLU activation function
        )
    def forward(self, x): #forward pass
        return self.block(x) #pass input through the block defined above in order
    

    #encoder block
class Downscale(nn.Module): #this block purpose: reduce resolution and increase feature channels to learn richer features
    def __init__(self, in_channels, out_channels): #initialize with input and output channels (it recieves in_channel feature maps, and outputs out_channel feature maps)
        super().__init__() #register parameters for pytorch

        # 
        self.block = nn.Sequential( # Sequential container to hold layers, runs in order
            nn.MaxPool2d(kernel_size=2), # Max pooling layer, kernel size 2 to reduce spatial dimensions by half so channels stay same, height and width are halved
            #identifies strong signals such as lines and edges, misses weak signals such as light shading
            DoubleConv(in_channels, out_channels) # Double convolution block to extract features from smaller images: more complex features can be learned
            #so each downscaling step doubles the number of feature channels, while halving the spatial dimensions
        )
    def forward(self, x): #forward pass
        return self.block(x) #pass input through the block defined above in order
    

    

    
    #decoder block
class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels): #initialize with input and output channels (it recieves in_channel feature maps, and outputs out_channel feature maps)
        super().__init__() #register parameters for pytorch

        # Define the upscaling block
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # Transposed convolution layer , expands the image size
        # stride 2 increases each pixel to a 2x2 block, effectively doubling height and width
        #channels will be reduced by half after upscaling
        self.conv = DoubleConv(in_channels, out_channels) # Double convolution block to refine features after upscaling

    def forward(self, x1, x2): #forward pass, takes two inputs: x1 (from previous layer) and x2 (from encoder for skip connection)
        x1  = self.up(x1) # Upscale x1 using transposed convolution, doubling its spatial dimensions, halving channels
            # Calculate the difference in dimensions between x1 and x2
        diffY = x2.size()[2] - x1.size()[2] # Height difference
        diffX = x2.size()[3] - x1.size()[3] # Width difference

            # Pad x1 to match the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2]) # Pad x1 to match x2 dimensions, add pixels on each side to align sizes in order left, right, top, bottom
            #divide by 2 so that pixels are added equally on both sides

        x = torch.cat([x2, x1], dim=1) # Concatenate x2 and x1 along the channel dimension (dim=1) (skip connection)
            #dim=1 means concatenate along channels, stack feature maps from both inputs
            #this helps retain high-resolution features lost during downscaling, while preserving context from upscaled features
        return self.conv(x) # Pass the concatenated tensor through the double convolution block to refine features
        


#full unet class using all blocks defined above
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):   #1 input channel (sketch), 3 output channels(RGB)
        super().__init__() #register parameters for pytorch
#image size is 256x256, so after 4 downscaling steps, smallest feature map is 16x16
        # Define the encoder path (downscaling) each block halves spatial dimensions and doubles feature channels
        self.convolution1 = DoubleConv(in_channels, 64) # First double convolution block
        self.down1 = Downscale(64, 128) # First downscaling block 
        self.down2 = Downscale(128, 256) # Second downscaling block
        self.down3 = Downscale(256, 512) # Third downscaling block
        self.down4 = Downscale(512, 1024) # Fourth downscaling block (bottleneck here)

        # Define the decoder path (upscaling)
        self.up1 = Upscale(1024, 512) # First upscaling block
        self.up2 = Upscale(512, 256) # Second upscaling block
        self.up3 = Upscale(256, 128) # Third upscaling block
        self.up4 = Upscale(128, 64) # Fourth upscaling block

        self.final_output = nn.Conv2d(64, out_channels, kernel_size=1) # Final 1x1 convolution to map to output channels (3, representing RGB)

    def forward(self, x: torch.Tensor) -> torch.Tensor: #forward pass
            # Encoder path
            conv = self.convolution1(x) # First double convolution
            firstDownscale = self.down1(conv) # First downscaling
            secondDownscale = self.down2(firstDownscale) # Second downscaling
            thirdDownscale = self.down3(secondDownscale) # Third downscaling
            bottleneck = self.down4(thirdDownscale) # Fourth downscaling (bottleneck)

            # Decoder path with skip connections
            upscale = self.up1(bottleneck, thirdDownscale) # First upscaling with skip connection from thirdDownscale 
            upscale = self.up2(upscale, secondDownscale) # Second upscaling with skip connection from secondDownscale
            upscale = self.up3(upscale, firstDownscale) # Third upscaling with skip connection from firstDownscale
            upscale = self.up4(upscale, conv) # Fourth upscaling with skip connection from first convolution output

            output = self.final_output(upscale) # Final output layer

            return torch.tanh(output) # Return the final output tensor, in [-1, 1] range for dataset matching
        
       

        

        
            

            


