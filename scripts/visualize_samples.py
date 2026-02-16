import cv2
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Temporary names — update later after confirming mapping
# CLASS_NAMES = ["cls0", "cls1", "cls2", "cls3", "cls4", "cls5"]
CLASS_NAMES = [
    "healthy",
    "caries",
    "impacted",
    "broken_root",
    "infection",
    "fractured"
]

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return x1, y1, x2, y2

def visualize_split(split="train", max_images=20):
    img_dir = Path("dataset/images") / split
    lbl_dir = Path("dataset/labels") / split
    out_dir = Path("artifacts/label_viz") / split
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS][:max_images]

    for img_path in images:
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)

            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imwrite(str(out_dir / f"{img_path.stem}_viz.jpg"), img)

    print(f"Saved visualizations to {out_dir}")

if __name__ == "__main__":
    visualize_split("train", 20)