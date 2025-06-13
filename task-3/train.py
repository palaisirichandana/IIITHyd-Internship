from ultralytics import YOLO

model = YOLO('yolo11n.pt')  
model.train(data="C:/Users/Shresth Agarwal/OneDrive/Documents/AIML training/Internship/task3/african-wildlife.yaml", epochs=15 , imgsz=640, batch=4, cache = True)
