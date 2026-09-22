from ultralytics import YOLO


MODEL = "yolo11n.pt"


model = YOLO(
    MODEL
)


results = model.train(

    data="datasets/retail_products/data.yaml",

    epochs=50,

    imgsz=640,

    batch=8,

    workers=2,

    project="runs/retail",

    name="retail_products"

)


print(
    "Training completed."
)


print(
    "Best model should be inside:"
)


print(
    "runs/retail/retail_products/weights/best.pt"
)