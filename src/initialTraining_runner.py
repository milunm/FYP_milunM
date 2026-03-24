from src.models.baseline_model import BaselineModel
from src.models.unet_model import UNet

from src.training_and_eval.train_supervised import train_supervised

if __name__ == "__main__":
    model =UNet() 

    train_supervised(
        model=model,
        run_name="UNET_Final",
        epochs=25,
        batch_size=3,
        lr=2e-4,
        save_debug_every=1,
       
    )
