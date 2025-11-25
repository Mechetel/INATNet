import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# === CONFIG SERVER ===
BASE_DIR = Path("~/INATNet/data").expanduser()
# BASE_DIR = Path("/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data").expanduser()
SRC_DIR = BASE_DIR / "GBRASNET"
DEST_DIR = BASE_DIR / "custom_big_png"

# dataset sizes
split_sizes = {
    "train": 5000,
    "val": 3000,
    "test": 2000,
}

# === FUNCTIONS ===
def collect_images(root_dir, type_name):
    """
    Collect all .pgm files recursively for 'cover' or 'stego'
    """
    all_images = list(root_dir.rglob("*.pgm"))
    return all_images


def convert_and_save(src_path, dest_path):
    """
    Convert .pgm → .png and save
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img.convert("L").save(dest_path, "PNG")


def prepare_subset(images, subset_name, type_name, count):
    """
    Take count images from list, convert, and save to target folder
    """
    subset_dir = DEST_DIR / subset_name / type_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    chosen = images[:count]
    for img_path in tqdm(chosen, desc=f"{subset_name}/{type_name}"):
        dest_file = subset_dir / (img_path.stem + ".png")
        convert_and_save(img_path, dest_file)


def main():
    # Collect cover and stego images
    print("Collecting image paths...")
    cover_images = collect_images(SRC_DIR, "cover")
    stego_images = collect_images(SRC_DIR, "stego")

    print(f"Found {len(cover_images)} cover and {len(stego_images)} stego images.")

    # Shuffle for randomness
    random.shuffle(cover_images)
    random.shuffle(stego_images)

    # Split according to sizes
    idx_train = split_sizes["train"]
    idx_val = idx_train + split_sizes["val"]
    idx_test = idx_val + split_sizes["test"]

    cover_splits = {
        "train": cover_images[:idx_train],
        "val": cover_images[idx_train:idx_val],
        "test": cover_images[idx_val:idx_test],
    }
    stego_splits = {
        "train": stego_images[:idx_train],
        "val": stego_images[idx_train:idx_val],
        "test": stego_images[idx_val:idx_test],
    }

    # Convert & copy
    for subset in ["train", "val", "test"]:
        prepare_subset(cover_splits[subset], subset, "cover", split_sizes[subset])
        prepare_subset(stego_splits[subset], subset, "stego", split_sizes[subset])

    print("\n✅ Dataset prepared successfully!")
    print(f"Saved to: {DEST_DIR}")


if __name__ == "__main__":
    main()
