from ultralytics import YOLO

model = YOLO('yolo11s.pt')  
model.train(data="/content/dataset.yaml", epochs=15 , imgsz=640, batch=4, cache = True)
#run the below code separately to predict
# to avoid re-training the model every time you run the script
model = YOLO("runs/detect/train/weights/best.pt")


results = model.predict(
    source="dataset/valid/images",  
    save=True                          
)
