import torch
import matplotlib.pyplot as plt
from src.models.baseline_model import BaselineModel
from src.models.unet_model import UNet
from src.DataLoader import test_loader
import random


device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda enabled rtx 3050 if available, else fallback to CPU

# LOAD MODELS
baseline = BaselineModel().to(device) #load baseline model and move to device
unet = UNet().to(device) #load UNET model and move to device
gan = UNet().to(device) #load GAN model and move to device

baseline.load_state_dict(torch.load("src/models/checkpoints/BaselineCNN_final/best_CNN_final.pt")["model_state"])
unet.load_state_dict(torch.load("src/models/checkpoints/UNET_final/best_unet_final.pt")["model_state"])
gan.load_state_dict(torch.load("src/models/checkpoints/GAN_FinalVersion/best_gan_final.pt")["generator_state"])
#load the best checkpoints (based on validation loss) for each model (switch to GAN generator state dict for GAN)
baseline.eval()
unet.eval()
gan.eval() # set all models to evaluation mode for inference (disables dropout, batch norm, etc)

# GET DATA
dataset = test_loader.dataset #get the dataset from the test data loader
indices = random.sample(range(len(dataset)), 3) #select 3 random indices
inputs = [] #empty lists to store inputs / targets
targets = []
for idx in indices:
    x, y = dataset[idx] #get images at random index
    inputs.append(x)
    targets.append(y) #append each input / target to list
input = torch.stack(inputs).to(device) #stack inputs into batch 
target = torch.stack(targets).to(device) #stack targets into batch

 #randomly select a single sample (input-target pair) from the test dataset for visualization

# PREDICT
with torch.no_grad():
    baseline_pred = baseline(input)
    unet_pred = unet(input)
    gan_pred = gan(input) #without gradients, forward pass through each model to get predictions for the input batch (baseline, UNET, GAN)

# FORMAT
def to_img(t):
    t = (t + 1) / 2 #convert from [-1,1] to [0,1] range for visualization
    return t.clamp(0,1).permute(0,2,3,1).cpu() #function to convert model outputs (tensors) to images for visualization, scales from [-1,1] to [0,1], permutes dimensions for plotting, and moves to CPU for matplotlib compatibility

input = to_img(input.repeat(1,3,1,1)) #repeat grayscale 3 times for viewing
target = to_img(target) #target
baseline_pred = to_img(baseline_pred)
unet_pred = to_img(unet_pred)
gan_pred = to_img(gan_pred) #convert input, target, and model predictions to images for visualization

# PLOT
batch_size = input.shape[0] #get the batch size from the input tensor shape to determine how many samples to visualize in the grid
titles = ["Input", "Baseline", "U-Net", "GAN", "Target"] #titles for the plot columns
fig, axes = plt.subplots(batch_size, 5, figsize=(12,3*batch_size)) # create a 5x3 grid of subplots for visualizing the results

for i in range(input.shape[0]): #loop through the number of samples:
    imgs = [input[i], baseline_pred[i], unet_pred[i], gan_pred[i], target[i]] #for each sample, create a list of the input image, baseline prediction, UNET prediction, GAN prediction, and target image for that sample
    for j in range(5): #loop through the 5 columns (input, baseline, UNET, GAN, target)
        axes[i,j].imshow(imgs[j]) #plot the corresponding image in the grid (input, baseline, UNET, GAN, target)
        axes[i,j].axis("off") #turn off axis for cleaner visualization
        if i == 0:
            axes[i,j].set_title(titles[j]) # set the title for the top row of the grid to indicate which column corresponds to which model/input/target

plt.tight_layout()
plt.savefig("compare_model_1.png", dpi=300, bbox_inches='tight') #save the plot as a high-resolution image file with tight bounding box to minimize whitespace
plt.show() #show plot