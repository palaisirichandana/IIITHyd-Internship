from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on an image from a URL
results = model('https://ultralytics.com/images/bus.jpg')

# Display the result in a window (this may not work in some terminal-only environments)
results[0].show()

# Save the result to a file
results[0].save(filename='detected_bus.jpg')
