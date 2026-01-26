from pathlib import Path #better paths than raw strings
from PIL import Image  #image loading
from torch.utils.data import Dataset  #pytorch dataset class
import torchvision.transforms as T  #image transformations

class PairedDataset(Dataset):
    def __init__(self, root="data/processed", split="train", size=256):
        self.root = Path(root)
        self.split = split #can be train, test, or val depending on input
        self.size = size #integer showing where the image will be resized to

        self.input_dir = self.root / split / "input" #path points to input folder
        self.target_dir = self.root / split / "target" #path points to target folder

        #collect all image (.png) file path objects in input and target directories, in sorted order (to maintain pairing)
        self.input_files = sorted(self.input_dir.glob("*.png"))
        self.target_files = sorted(self.target_dir.glob("*.png"))

        



        #safety checks

        if len(self.input_files) != len(self.target_files):
            raise ValueError(f"Number of input and target images do not match in {split} set")
        #ensure input and target files are the same length

        if len(self.input_files) == 0: #check if images are present in directory
            raise ValueError(f"No images found in {self.input_dir}")
        
        #transforms to be applied to both input and target images
        self.sketch_transform = T.Compose([ #apply transformations in order:
            T.Resize((self.size, self.size)), #resize to size x size (as defined in parameters) e.g 256 x 256 ensures all data is same size
            T.ToTensor(), #(convert to PyTorch tensor)
        ])
        self.color_transform = T.Compose([
            T.Resize((self.size, self.size)), #same as sketch transform 
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_files) #return number of image pairs in dataset
 
    def __getitem__(self, idx):
        # get file paths for this specific index
        sketch_path = self.input_files[idx]
        color_path = self.target_files[idx]

        #load images, force correct color modes
        sketch_img = Image.open(sketch_path).convert("L")  #sketch is in L color space (1 channel)
        color_img = Image.open(color_path).convert("RGB")  #color image is in RGB color space(3 channels)
        
        #apply transforms (resize and convert to tensor)
        sketch_tensor = self.sketch_transform(sketch_img)
        color_tensor = self.color_transform(color_img)

        # normalize images to [-1, 1] range (from initial [0,1] range
        sketch_tensor = sketch_tensor * 2 - 1
        color_tensor = color_tensor * 2 - 1
 
        return sketch_tensor, color_tensor  #return the paired tensors for training
    

  