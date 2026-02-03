from src.models.baseline_model import BaselineModel
from src.models.unet_model import UNet

from src.training_and_eval.train_supervised import train_supervised

if __name__ == "__main__":
    model = BaselineModel()

    train_supervised(
        model=model,
        run_name="BaselineCNN",
        epochs=10,
        batch_size=8,
        lr=2e-4,
        save_debug_every=1,
        test_each_epoch=False  # keep False; run test once at the end
    )
