# import torch
# from ultralytics import YOLO

# def pick_device() -> str:
#     # Apple Silicon GPU (MPS)
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"

# if __name__ == "__main__":
#     device = pick_device()
#     print(f"Using device: {device}")

#     model = YOLO("yolov8n.pt")
#     model.train(
#         data="dataset/data.yaml",
#         epochs=60,
#         imgsz=640,
#         batch=4,          # keep small on laptop
#         device=device,    # "mps" or "cpu"
#         project="runs",
#         name="dentassist_yolo"
#     )

import torch
from ultralytics import YOLO

def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    device = pick_device()
    print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=512,
        batch=2,
        device=device,
        workers=0,
        project="runs",
        name="dentassist_yolo"
    )