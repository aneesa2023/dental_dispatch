import json
from pathlib import Path
from ultralytics import YOLO
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png"}
# CLASS_NAMES = ["cls0", "cls1", "cls2", "cls3", "cls4", "cls5"]
CLASS_NAMES = [
    "healthy",
    "caries",
    "impacted",
    "broken_root",
    "infection",
    "fractured"
]

def run_split(model, split):
    img_dir = Path("dataset/images") / split
    out_overlay = Path("artifacts/inference") / split / "overlays"
    out_json = Path("artifacts/inference") / split / "reports"

    out_overlay.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    for img_path in images:
        result = model.predict(str(img_path), conf=0.25, verbose=False)[0]

        # Save overlay
        overlay = result.plot()
        overlay_path = out_overlay / f"{img_path.stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        findings = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                findings.append({
                    "label_id": cls_id,
                    "label": CLASS_NAMES[cls_id],
                    "confidence": float(box.conf.item())
                })

        report = {
            "image": img_path.name,
            "overlay": overlay_path.name,
            "findings": findings
        }

        json_path = out_json / f"{img_path.stem}.json"
        json_path.write_text(json.dumps(report, indent=2))

    print(f"Inference completed for {split}")

if __name__ == "__main__":
    model = YOLO("runs/detect/runs/dentassist_yolo2/weights/best.pt")

    for split in ["train", "val", "test"]:
        run_split(model, split)

    print("All done.")