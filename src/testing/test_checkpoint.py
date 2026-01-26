import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from pathlib import Path


# for UNET, switch commented lines
from src.models.baseline_model import BaselineModel as ModelClass
# from src.models.unet_model import UNetModel as ModelClass   # <- later

from src.dataset import PairedDataset  #import the PairedDataset class from dataset.py

# data loader import for testing 
from src.DataLoader import test_loader


# choose checkpoint here 
CKPT_PATH = "src/models/checkpoints/baseline/best_baseline.pt"



OUT_DIR = "src/testing/outputs"     #output directory for test results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #prioritize GPU if available


def main():
    device = torch.device(DEVICE)

    # load model
    model = ModelClass().to(device) #load model class and move to device
    model.eval() #eval mode for inference

    # load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu",weights_only=True)

   
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    # load test data from import
    loader = test_loader

    l1 = nn.L1Loss() #initialize L1 loss function (mean absolute error)
    total = 0.0 #total loss accumulator, starting at 0
    n = 0 #number of batches processed, starting at 0

    out_dir = Path(OUT_DIR) / Path(CKPT_PATH).stem  #create output directory path based on checkpoint name
    out_dir.mkdir(parents=True, exist_ok=True) #make the output directory if it doesn't exist

    saved_grid = False #boolean flag to track if the output grid has been saved

    with torch.no_grad(): #disable gradient calculations for test
        for x, y in loader: #iterate through each batch in the test data loader
            x = x.to(device) #move input batch to device
            y = y.to(device)# move target batch to device

            pred = model(x) #get model predictions for input batch

            loss = l1(pred, y) #compute L1 loss between predictions and ground truth
            total += float(loss.item()) #accumulate loss value to total
            n += 1 #increment batch count by 1

            # save ONE visual grid (input | pred | target) from the first batch only
            if not saved_grid:
                x_vis = x.clamp(0, 1).cpu() #clamp input to [0,1] and move to CPU for saving
                p_vis = pred.clamp(0, 1).cpu() #clamp prediction to [0,1] and move to CPU for saving
                y_vis = y.clamp(0, 1).cpu() #clamp target to [0,1] and move to CPU for saving

                if x_vis.size(1) == 1:  # if input is single channel (grayscale)
                    x_vis = x_vis.repeat(1, 3, 1, 1)  # convert to 3-channel by repeating
                #should be changed from [-1,1] to [0,1] for better viewing of predictions and images

                stacked = torch.cat([x_vis, p_vis, y_vis], dim=0) #stack input, prediction, and target vertically
                grid = make_grid(stacked, nrow=x_vis.size(0), padding=4) #create image grid with specified number of rows and padding
                save_image(grid, out_dir / "grid_first_batch.png") #save the image grid to output directory
                saved_grid = True #set flag to True to avoid saving multiple grids

    avg_l1 = total / max(1, n) #compute average L1 loss over all batches
    (out_dir / "metrics.txt").write_text(f"avg_L1: {avg_l1:.6f}\n") #save average L1 loss to metrics text file in output directory

    print("Checkpoint:", CKPT_PATH)
    print(f"Test avg L1: {avg_l1:.6f}")
    print("Saved:", str(out_dir)) # final print statements summarizing results


if __name__ == "__main__": #only run if this script is executed directly, not imported
    main() #run the main function
