from src.training_and_eval.train_gan import train_GAN
from src.models.unet_model import UNet
from src.models.discriminator import Discriminator

if __name__ == "__main__":
    train_GAN(
        run_name="GAN_FinalVersion",
        epochs=25,
        batch_size=3,
        lr=2e-4,
        save_debug_every=1,

    )