import os
import argparse
from ultralytics import YOLO

def train_model(
    data_yaml: str = "dataset/data.yaml",
    model_size: str = "yolov8s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16
):
    """
    Train a custom YOLOv8 model for PPE detection.
    Can be run locally or uploaded to Google Colab for free GPU training.
    """
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Could not find dataset config at: {data_yaml}")
        print("Please export your annotated dataset from Roboflow in 'YOLOv8' format and place it in the project root.")
        return

    print("=" * 60)
    print("STARTING PPE YOLOV8 CUSTOM TRAINING")
    print(f"Dataset Config : {data_yaml}")
    print(f"Base Model     : {model_size}")
    print(f"Epochs         : {epochs}")
    print(f"Image Size     : {imgsz}")
    print(f"Batch Size     : {batch}")
    print("=" * 60)

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        plots=True,
        save=True,
        project="runs/train",
        name="ppe_custom_model"
    )

    print("\n[TRAINING FINISHED]")
    print(f"Best weights saved at: runs/train/ppe_custom_model/weights/best.pt")
    print("Copy 'best.pt' into 'models/ppe_custom.pt' to use it in your surveillance system!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 PPE model")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model (yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
