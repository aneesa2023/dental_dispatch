import zipfile
from pathlib import Path
import shutil

ZIP_PATH = Path("data_raw/archive.zip")

# Inside your zip, the object detection split lives here:
# Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Augmented Dataset/{train,valid,test}/{images,labels}
OD_ROOT = Path("Dental OPG XRAY Dataset") / "Dental OPG (Object Detection)" / "Augmented Dataset"

SPLIT_MAP = {
    "train": "train",
    "valid": "val",   # convert "valid" to "val"
    "test": "test",
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found at: {ZIP_PATH.resolve()}")

    extract_dir = Path("data_raw/extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {ZIP_PATH} -> {extract_dir}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_dir)

    # Source base path after extraction
    src_base = extract_dir / OD_ROOT

    if not src_base.exists():
        raise FileNotFoundError(
            f"Expected object detection folder not found:\n{src_base}\n"
            "Check your zip contents and path constants."
        )

    dst_base = Path("dataset")
    for split_src, split_dst in SPLIT_MAP.items():
        src_images = src_base / split_src / "images"
        src_labels = src_base / split_src / "labels"

        dst_images = dst_base / "images" / split_dst
        dst_labels = dst_base / "labels" / split_dst
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying split '{split_src}' -> '{split_dst}'")
        # Copy images
        for img in src_images.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                shutil.copy2(img, dst_images / img.name)

        # Copy labels
        for lbl in src_labels.iterdir():
            if lbl.suffix.lower() == ".txt":
                shutil.copy2(lbl, dst_labels / lbl.name)

        print(f"  Images: {len(list(dst_images.iterdir()))}")
        print(f"  Labels: {len(list(dst_labels.iterdir()))}")

    print("\nDone. Dataset prepared at ./dataset")

if __name__ == "__main__":
    main()