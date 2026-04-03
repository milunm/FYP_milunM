import torch #pytorch core library
import torch.nn as nn #pytorch neural network module
from torch.utils.data import DataLoader #pytorch data loader class to make batches
from torchvision.utils import save_image #utility to save images and verify outputs

from src.dataset import PairedDataset  #import the PairedDataset class from dataset.py (creates paired tensors)
from src.models.baseline_model import BaselineModel  #import the BaselineModel CNN model

def save_debug_grid(x, pred, y, path, nrow=4):
    """
    Saves a grid with 3 rows:
      Row 1: input sketch
      Row 2: model prediction
      Row 3: ground truth target
    """
    # Convert from [-1,1] to [0,1]
    x_vis = (x + 1) / 2
    pred_vis = (pred + 1) / 2
    y_vis = (y + 1) / 2

    # Sketch is 1-channel; repeat to 3 channels for viewing
    x_vis = x_vis.repeat(1, 3, 1, 1)

    # Stack into one big batch: [inputs, preds, targets]
    grid = torch.cat([x_vis, pred_vis, y_vis], dim=0)

    # Save as image grid
    save_image(grid.clamp(0, 1), path, nrow=x.size(0))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use GPU if available, else CPU
print(f"Using device: {device}") #print which device is being used

#create dataset object pointing at training input and target folders
train_dataset = PairedDataset(root="data/processed", split="train", size=256)
#can return one tensor pair at a time
#input shape (1, H, W), target shape (3, H, W)

#Wrap dataset in a DataLoader to create batches and shuffle data
loader = DataLoader(train_dataset, batch_size=20, shuffle=True, pin_memory=True, num_workers=0)
#each batch contains 20 pairs, shuffling ensures different order each epoch

#initialize the baseline model and move it to the selected device (GPU/CPU)
model = BaselineModel().to(device)

#loss function and optimizer

loss_fn = nn.L1Loss()
#using L1 loss, mean absolute error between prediction and ground truth
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#adam optimizer, updates model weights based on computed gradients. lr is the learning rate

#training loop

for epoch in range(15):

    model.train()  #set model to training mode

    #loop through each batch in the data loader
    for x, y in loader:
        x = x.to(device)  #move input batch to device
        y = y.to(device)  #move target batch to device (cpu or gpu)

        optimizer.zero_grad()  #clear previous gradients from past steps

        prediction = model(x)  #forward pass: get model predictions for input batch

        loss = loss_fn(prediction, y)  #compute loss between predictions and ground truth

        loss.backward()  #backpropagate to compute gradients

        optimizer.step()  #update model weights based on gradients

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  #print epoch averagepyt loss value to 4 dp

#save some example outputs after training to verify results
model.eval()  #set model to evaluation mode

x_batch, y_batch = next(iter(loader))  #get a single batch from the training loader
x_batch, y_batch = x_batch.to(device), y_batch.to(device)

with torch.no_grad():
    pred = model(x_batch)

#clamp predictions within normalized range only for saving
pred_vis = pred.clamp(-1, 1)

# Convert from [-1,1] -> [0,1] for image saving
pred_vis = (pred_vis + 1) / 2

# Sketch + target for comparison
x_vis = (x_batch + 1) / 2
y_vis = (y_batch + 1) / 2
x_vis = x_vis.repeat(1, 3, 1, 1)

grid = torch.cat([x_vis, pred_vis, y_vis], dim=0)
save_image(grid, "debug_epoch01.png", nrow=x_batch.size(0))

print("Saved debug_epoch01.png")




print("pred range:", float(pred.min()), float(pred.max()))








