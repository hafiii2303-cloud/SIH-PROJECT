from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "traindatasets"
    / "retailproduct_v3"
    / "data.yaml"
)

MODEL_PATH = "yolo11n.pt"

print("Dataset:")
print(DATASET_PATH)

print("Dataset exists:", DATASET_PATH.exists())

model = YOLO(MODEL_PATH)

results = model.train(
    data=str(DATASET_PATH),
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    project=str(BASE_DIR / "runs" / "retail"),
    name="retail_products_v3"
)

print("Training completed!")