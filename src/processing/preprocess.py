import random
from pathlib import Path
from PIL import Image

RAW_ROOT = Path("data/raw/AnitaDataset") #raw unprocessed anita dataset (input)
OUT_ROOT = Path("data/processed") #processed dataset output location 

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1} #80% train, 10% val, 10% test
SEED = 42 #control randomness for reproducibility


def is_image(name: str) -> bool:
    name = name.lower() #lowercase so PNG or png work
    return name.endswith((".png", ".jpg", ".jpeg", ".webp")) #return true only if the filename is a valid image extension


def ensure_dirs():
    for split in SPLITS:
        (OUT_ROOT / split / "input").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "target").mkdir(parents=True, exist_ok=True) #create input and target directories for all 3 splits


def collect_pairs():
    pairs = [] #initialise empty list to hold valid pairs
    projects = [p for p in RAW_ROOT.iterdir() if p.is_dir()] #find all project folders in raw data (there are lots of diff 'projects' with sketch/color pairs within the full dataset)
    print(f"Found {len(projects)} project folders")

    for proj in sorted(projects): #iterate through each project folder in sorted order
        sketch_root = proj / "sketch" #path to sketch folder
        color_root = proj / "color" #path to color folder

        if not sketch_root.exists() or not color_root.exists():
            print(f"Skipping {proj.name} (missing sketch/color)")
            continue #if any project is missing either folder, skip it

        for sketch_path in sketch_root.rglob("*"): #for every subolder and file inside the sketch folder
            if not sketch_path.is_file() or not is_image(sketch_path.name): #only keep valid image files
                continue

            rel = sketch_path.relative_to(sketch_root) #get relative path of sketch file within sketch folder
            color_path = color_root / rel #construct corresponding color image path using same relative path (as they have same relative path, just diff root either sketch or color)

            if color_path.exists(): #if the same path exists in color folder, append the 2 images as a pair
                pairs.append((sketch_path, color_path))

    print(f"Collected {len(pairs)} valid pairs")
    return pairs #return all pairs found


def split_pairs(pairs):
    random.seed(SEED) #reproducible random seed
    random.shuffle(pairs) #shuffle the pairs randomly

    n = len(pairs) #length of all pairs
    n_train = int(n * SPLITS["train"]) #number of training samples (total multiplied by train percentage)
    n_val = int(n * SPLITS["val"]) #number of validation samples (total multiplied by val percentage)
 #slice shuffled list into three pieces by index
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    return {"train": train, "val": val, "test": test} #return the three splits


def save_split(split_name, pairs):
    in_dir = OUT_ROOT / split_name / "input" 
    tg_dir = OUT_ROOT / split_name / "target" #set input and target paths based on the inputted splits

    saved = 0 #track number of saved pairs
    for idx, (sketch_path, color_path) in enumerate(pairs): #loop through each pair in the split, generate an index
        try: #try-except block to catch errors during saving
            with Image.open(sketch_path) as s: #open sketch file from raw dataset
                s = s.convert("L") #convert to L mode (1 channel - grayscale), aligning with model expectation
                s.save(in_dir / f"{idx:06d}.png") #save in input directory with index filename padded with 6 zeros

            with Image.open(color_path) as c: #open color file from raw dataset
                c = c.convert("RGB") #convert to RGB mode (3 channel - color), aligning with model expectation
                c.save(tg_dir / f"{idx:06d}.png") #save in target directory with index filename padded with 6 zeros

            saved += 1 #increment saved count
            if saved % 500 == 0:
                print(f"[{split_name}] saved {saved} pairs...") #feedback every 500 pairs saved

        except Exception as e:
            print(f"[{split_name}] Failed: {sketch_path} ({e})") #any error, print and continue

    print(f"{split_name}: saved {saved} pairs") #at the end print total saved for this split


def main(): #run entire preprocess pipeline
    ensure_dirs() #create necessary directories for splits, or do nothing if they already exist

    pairs = collect_pairs() #scan raw data, find valid pairs, store them as a list in memory
    splits = split_pairs(pairs) #shuffle and split pairs into train/val/test dictionaries

    for split_name, split_list in splits.items():   #for eachsplit, save images to disk in the correct folders
        save_split(split_name, split_list)

    print("\nDataset split complete.")
    for k, v in splits.items():
        print(f"{k}: {len(v)} samples") #print samples in each split


if __name__ == "__main__":
    main() #only run main function if the file is directly executed
