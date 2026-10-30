from ultralytics import YOLO

model = YOLO(r"C:\Users\dell\runs\detect\train-6\weights\best.pt")

print(model.names)