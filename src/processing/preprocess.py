from pathlib import Path #better paths than raw strings
import cv2 # OpenCV for image processing (resizing, color space conversion)
from PIL import Image # for saving images (PIL can handle saving in various formats easily)
import random # for shuffling dataset before splitting

# -----------------------------
# Configuration
# -----------------------------

RAW_ROOT = Path("data/leftImg8bit_trainvaltest/leftImg8bit") # root directory containing raw images, expected to have subfolders for train/val/test with city subfolders inside
OUT_ROOT = Path("data/processed") # output directory for processed images, will have subfolders train/val/test each containing input and target folders

SIZE = (256, 256) #predefined image dimensions for resizing (width, height)

# reproducible shuffle
random.seed(42)

# -----------------------------
# Create output folders
# -----------------------------

splits = ["train", "val", "test"] # create input and target subfolders for each split (train/val/test)

for split in splits:
    (OUT_ROOT / split / "input").mkdir(parents=True, exist_ok=True) #iterate and create input folders
    (OUT_ROOT / split / "target").mkdir(parents=True, exist_ok=True) #iterate and create target folders

# -----------------------------
# Collect all images
# -----------------------------

image_paths = list(RAW_ROOT.rglob("*.png")) # recursively find all .png images in RAW_ROOT (and subdirectories), returns a list of Path objects
print(f"Total images found: {len(image_paths)}") # print total number of images found, should be 5000   for the leftImg8bit dataset

random.shuffle(image_paths)

# -----------------------------
# Split dataset (80 / 10 / 10)
# -----------------------------

total = len(image_paths) #get total number of images found to calculate split indexes for train/val/test splits based on 80/10/10 ratio

train_end = int(0.8 * total) #index for end of training split (80% of total)
val_end = int(0.9 * total) #index for end of validation split (90% of total, since validation is 10% after training)

train_files = image_paths[:train_end] # assign the first 80% of the shuffled image paths to the training split, this will be used for training the model
val_files = image_paths[train_end:val_end] # assign the next 10% of the shuffled image paths to the validation split, this will be used for tuning hyperparameters and early stopping during training
test_files = image_paths[val_end:]  # assign the last 10% of the shuffled image paths to the test split, this will be used for final evaluation of the model after training is complete

dataset_splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files #store the file paths for each split in a dictionary for easy access during processing
}

# -----------------------------
# Process images
# -----------------------------

for split, files in dataset_splits.items(): #iterate through each split and its corresponding file paths

    print(f"Processing {split} ({len(files)} images)...")

    for img_path in files:

        # load image (OpenCV loads BGR)
        bgr = cv2.imread(str(img_path)) # read image from disk using OpenCV, returns a numpy array in BGR color space (3 channels)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # convert from BGR to RGB color space, since OpenCV uses BGR by default but we want RGB for our dataset and model training

        # resize image
        rgb = cv2.resize(rgb, SIZE) # resize the RGB image to the predefined SIZE (256x256) using OpenCV's resize function, this ensures all images are the same size for training

        # convert RGB -> LAB
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB) # convert the resized RGB image to LAB color space using OpenCV, LAB separates lightness (L) from color information (A and B channels), which is useful for our task of colorization 

        # extract L channel, this will be our input (grayscale image) for the model, it has values in the range [0, 255]
        L = lab[:, :, 0] # extract the L channel (lightness) from the LAB image, this will be our input (grayscale sketch) for the model, it has values in the range [0, 255]

        filename = img_path.name # get the filename from the original image path, we will use this to save the processed input and target images with the same name in their respective folders (input and target) for easy pairing during training

        # save grayscale input
        input_path = OUT_ROOT / split / "input" / filename # define the path to save the grayscale input image, it will be saved in the input folder of the corresponding split (train/val/test) with the same filename as the original image
        Image.fromarray(L).save(input_path) # save the L channel as a grayscale image using PIL, this will be our input for the model during training, it is saved in the input folder of the corresponding split (train/val/test) with the same filename as the original image

        # save RGB target
        target_path = OUT_ROOT / split / "target" / filename # define the path to save the RGB target image, it will be saved in the target folder of the corresponding split (train/val/test) with the same filename as the original image
        Image.fromarray(rgb).save(target_path) # save the resized RGB image as the target using PIL, this will be our target for the model during training, it is saved in the target folder of the corresponding split (train/val/test) with the same filename as the original image

print("Preprocessing complete.")