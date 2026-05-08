from ultralytics import YOLO

# 1. Load pre-built architecture
model = YOLO('yolov8n.pt')

# 2. Start training
# Make sure to provide the full path to your data.yaml file
model.train(
    data=r"C:\Users\baris\Desktop\medical\data\data.yaml",
    epochs=25,     # Number of passes through the dataset (25 is ideal for starting)
    imgsz=640,     # Image size
    device='cpu'   # Use '0' if you have an NVIDIA GPU, otherwise keep 'cpu'
)
