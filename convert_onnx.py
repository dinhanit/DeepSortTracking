from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("weight/yolov8n.pt")
    model.export(format="onnx",half=True)
