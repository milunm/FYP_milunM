import os
from pathlib import Path #better file paths

import torch #main pytorch package
import torch.nn as nn #neural network module
from torch.utils.data import DataLoader #data loader class to make batches
from torchvision.utils import save_image #utility to save images and verify outputs

from src.dataset import PairedDataset #import my paired dataset class that pairs tensors


def save_debug_grid(x, pred, y, out_path: Path): #function to save debug image grids
  
    out_path.parent.mkdir(parents=True, exist_ok=True) #will create parent directories if they don't exist

    pred = pred.clamp(-1, 1) #clamp predictions to [-1, 1] range, matching dataset normalization

    # denorm [-1,1] -> [0,1] for image display, to avoid dark and unnatural images 
    x_vis = (x + 1) / 2
    y_vis = (y + 1) / 2
    p_vis = (pred + 1) / 2

    # sketch is 1-channel -> repeat to 3 channels for viewing
    x_vis = x_vis.repeat(1, 3, 1, 1)

    grid = torch.cat([x_vis, p_vis, y_vis], dim=0) #stack into one big batch: [inputs, predicted,targets]
    save_image(grid.clamp(0, 1), str(out_path), nrow=x.size(0)) #save as image grid


@torch.no_grad() #disable gradient calculations for evaluation
def evaluate_epoch(model, loader, loss_fn, device): #inputs model, data loader, loss function, device
    """
    Runs evaluation on a loader (val or test).
    Returns average loss over all batches.
    """
    model.eval() #set model to evaluation mode
    total = 0.0 #initialize total loss to 0

    for x, y in loader: #for each batch in the data loader
        x = x.to(device)
        y = y.to(device) #move input and target batches to device (cpu or gpu)

        pred = model(x) #forward pass: get model predictions
        loss = loss_fn(pred, y) #compute loss between predictions and ground truth
        total += loss.item() #accumulate loss

    return total / max(1, len(loader)) #return average loss over all batches


def train_supervised( #inputs for supervised training function
    model, #model (e.g BaselineModel)
    run_name: str, #name for this training run (used for saving checkpoints)
    epochs: int = 10, #number of training epochs CAN B easily changed
    lr: float = 2e-4, #learning rate for optimizer
    batch_size: int = 8, #batch size for data loader
    size: int = 256, #image size
    num_workers: int = 0, #number of worker threads for data loading (set to 0 as issues occured with higher numbers on Windows)
    pin_memory: bool = True, #pin memory for faster data transfer to GPU
    save_debug_every: int = 5,#save debug grid every N epochs
    test_each_epoch: bool = False,   # whether to run test set evaluation each epoch
):
    """
    Supervised trainer for sketch->color models.
    - trains on train split
    - runs validation each epoch
    
    
    """

    # device prioritization: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device) #print which device is being used

    # -------- data --------
    train_ds = PairedDataset(split="train", size=size) #create dataset object pointing at training input and target folders
    val_ds = PairedDataset(split="val", size=size) #create dataset object pointing at validation input and target folders
    test_ds = PairedDataset(split="test", size=size) #create dataset object pointing at test input and target folders

    train_loader = DataLoader( #data loader for training batches and shuffle data
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader( #data loader for validation batches without shuffling
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader( #data loader for test batches without shuffling
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory 
    )   #data loaders for train, val, test sets

    # -------- model --------
    model = model.to(device) #move model to the selected device (GPU/CPU)

    # -------- loss + optimizer --------
    loss_fn = nn.L1Loss() #using L1 loss, mean absolute error between prediction and ground truth
    opt = torch.optim.Adam(model.parameters(), lr=lr) #adam optimizer, updates model weights based on computed gradients. lr is the learning rate

    # -------- checkpoint folder --------
    ckpt_dir = Path("src/models/checkpoints") / run_name #create checkpoint directory path with run name
    ckpt_dir.mkdir(parents=True, exist_ok=True) #create the checkpoint directory if it doesn't exist

    best_val = float("inf") #initialize best validation loss to infinity

    for epoch in range(1, epochs + 1): #training loop for specified number of epochs
        # ===== TRAIN =====
        model.train()
        train_total = 0.0 

        for x, y in train_loader: #for each batch in the training data loader
            x = x.to(device)
            y = y.to(device) #move input and target batches to device (cpu or gpu)

            pred = model(x)
            loss = loss_fn(pred, y) #forward pass: get model predictions and compute loss

            opt.zero_grad() #clear previous gradients from past steps
            loss.backward() #backpropagate to compute gradients
            opt.step() #update model weights based on gradients and learning rate

            train_total += loss.item() #accumulate training loss

        train_avg = train_total / max(1, len(train_loader)) #compute average training loss over all batches

        #Validate every epoch using above defined function. also prints val loss
        val_avg = evaluate_epoch(model, val_loader, loss_fn, device) #compute average validation loss over all batches

        #optional to run test set evaluation each epoch
        if test_each_epoch:
            test_avg = evaluate_epoch(model, test_loader, loss_fn, device)
            print(
                f"Epoch {epoch:02d}/{epochs} | train L1: {train_avg:.4f} | val L1: {val_avg:.4f} | test L1: {test_avg:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs} | train L1: {train_avg:.4f} | val L1: {val_avg:.4f}"
            )

        # SAVE last checkpoint (every epoch) 
        torch.save( 
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "train_l1": train_avg,
                "val_l1": val_avg,
            },
            ckpt_dir / "last.pt" #save last checkpoint
        )

        #  SAVE best checkpoint (based on val) 
        if val_avg < best_val:
            best_val = val_avg
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "best_val_l1": best_val,
                },
                ckpt_dir / "best.pt" #save best checkpoint
            )
            print(" saved best.pt")

        # SAVE debug images depending on inputted frequency

        if epoch == 1 or (save_debug_every and epoch % save_debug_every == 0): #every N epochs as inputted
            model.eval()
            x_vis, y_vis = next(iter(val_loader))
            x_vis = x_vis.to(device)
            y_vis = y_vis.to(device) #get a single batch from the validation loader and move to device

            with torch.no_grad():
                pred_vis = model(x_vis) #get model predictions for this batch, without gradient calculations as it's eval

            save_debug_grid( # save debug image grid function defined above
                x_vis, pred_vis, y_vis,
                ckpt_dir / f"debug_epoch{epoch:02d}.png"
            )
            print(f"Saved debug_epoch{epoch:02d}.png")

    print(f"Done. Checkpoints saved in: {ckpt_dir}") #final print statement with checkpoint directory
