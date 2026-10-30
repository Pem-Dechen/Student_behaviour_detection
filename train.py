from ultralytics import YOLO

# =========================
# FAST TRAINING MODE
# =========================

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\dell\Downloads\final project\student detection.v3i.yolov8\data.yaml",
    epochs=10,        #  reduced for speed
    imgsz=416,       # smaller image size = faster
    batch=4,         # lower batch for CPU
    device="cpu",
    workers=2
)

print("FAST TRAINING COMPLETED")