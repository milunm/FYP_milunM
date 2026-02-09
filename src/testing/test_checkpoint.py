import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from pathlib import Path

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure 
import lpips #metrics for analysing results


# for UNET, switch commented lines
from src.models.baseline_model import BaselineModel as ModelClass
#from src.models.unet_model import UNet as ModelClass   # easy switch between models, GAN is same type



from src.dataset import PairedDataset  #import the PairedDataset class from dataset.py

# data loader import for testing 
from src.DataLoader import test_loader


# choose checkpoint here 
#CKPT_PATH = "src/models/checkpoints/UNET/best_UNET.pt"
#CKPT_PATH = "src/models/checkpoints/GAN_Pix2Pix/last_gan.pt" #for GAN, switch to GAN checkpoint
CKPT_PATH = "src/models/checkpoints/BaselineCNN/best_baseline.pt" #for baseline model, switch to baseline checkpoint



OUT_DIR = "src/testing/outputs"     #output directory for test results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #prioritize GPU if available


def main():
    device = torch.device(DEVICE)

    # load model
    model = ModelClass().to(device) #load model class and move to device
    model.eval() #eval mode for inference

    # load checkpoint
    

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)  # Added weights_only=True to load only the model weights, ignoring any optimizer state or other metadata in the checkpoint
    #start at CPU for loading to avoid GPU memory issues, then move model to device after loading weights

    print("CKPT type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("CKPT keys:", list(ckpt.keys())[:30])


   
    state_dict = ckpt["model_state"] 
    #state_dict = ckpt["generator_state"] #for GAN, switch to generator state dict
    model.load_state_dict(state_dict, strict=True)
    model.to(device)  #move model to device

    # load test data from import
    loader = test_loader

    l1 = nn.L1Loss() #initialize L1 loss function (mean absolute error)
    total = 0.0 #total loss accumulator, starting at 0
    n = 0 #number of batches processed, starting at 0

    #metrics intialised
    PeakSignalNoiseRatio_metric = PeakSignalNoiseRatio(data_range=2.0).to(device) #PSNR metric for image quality assessment, data range set to 2 for [-1,1] images
    StructuralSimilarityIndexMeasure_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device) #SSIM metric for image similarity, data range set to 2 for [-1,1] images
    lpips_metric = lpips.LPIPS(net="alex").to(device) #LPIPS metric for perceptual similarity, using AlexNet backbone

    #metric accumulators
    total_psnr = 0.0 #total PSNR accumulator
    total_ssim = 0.0 #total SSIM accumulator
    total_lpips = 0.0 #total LPIPS accumulator

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
            #metric calculations and accumulations
            psnr_value = PeakSignalNoiseRatio_metric(pred, y) #compute PSNR for current batch
            ssim_value = StructuralSimilarityIndexMeasure_metric(pred, y) #compute SSIM for current batch
            lpips_value = lpips_metric(pred, y).mean() #compute LPIPS for current batch
            total_psnr += float(psnr_value.item()) #accumulate PSNR value to total
            total_ssim += float(ssim_value.item()) #accumulate SSIM value to total
            total_lpips += float(lpips_value.item()) #accumulate LPIPS value to total

            # save ONE visual grid (input | pred | target) from the first batch only
            if not saved_grid:

                x_vis = ((x + 1) / 2).clamp(0, 1).cpu() #convert from [-1,1] to [0,1] and move to CPU for saving
                p_vis = ((pred + 1) / 2).clamp(0, 1).cpu() #convert from [-1,1] to [0,1] and move to CPU for saving
                y_vis = ((y + 1) / 2).clamp(0, 1).cpu() #convert from [-1,1] to [0,1] and move to CPU for saving

                if x_vis.size(1) == 1:  # if input is single channel (grayscale)
                    x_vis = x_vis.repeat(1, 3, 1, 1)  # convert to 3-channel by repeating
                #should be changed from [-1,1] to [0,1] for better viewing of predictions and images

                stacked = torch.cat([x_vis, p_vis, y_vis], dim=0) #stack input, prediction, and target vertically
                grid = make_grid(stacked, nrow=x_vis.size(0), padding=4) #create image grid with specified number of rows and padding
                save_image(grid, out_dir / "grid_first_batch.png") #save the image grid to output directory
                saved_grid = True #set flag to True to avoid saving multiple grids

    avg_l1 = total / max(1, n) #compute average L1 loss over all batches
    avg_psnr = total_psnr / max(1, n) #compute average PSNR over all batches
    avg_ssim = total_ssim / max(1, n) #compute average SS
    avg_lpips = total_lpips / max(1, n) #compute average LPIPS over all batches

    (out_dir / "metrics.txt").write_text(f"avg_L1: {avg_l1:.6f}\n"
                                             f"avg_PSNR: {avg_psnr:.6f}\n"
                                             f"avg_SSIM: {avg_ssim:.6f}\n"
                                             f"avg_LPIPS: {avg_lpips:.6f}\n"
                                      
    ) #save average L1 loss and other metrics to metrics text file in output directory
  

    print("Checkpoint:", CKPT_PATH)
    print(f"Test avg L1: {avg_l1:.6f}")
    print(f"Test avg PSNR: {avg_psnr:.6f}")
    print(f"Test avg SSIM: {avg_ssim:.6f}")
    print(f"Test avg LPIPS: {avg_lpips:.6f}")
    print("Saved:", str(out_dir)) # final print statements summarizing results


if __name__ == "__main__": #only run if this script is executed directly, not imported
    main() #run the main function
