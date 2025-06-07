from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

image_paths = [
    'input_files/image1.jpg',
    'input_files/image2.jpg',
    'input_files/image3.jpg',
    'input_files/image4.jpg'
]

# Loop through each image
for img_path in image_paths:
    results = model(img_path)
    
    # Save segmented output (same filename prefixed with 'seg_')
    output_path = 'output_segmented/seg_' + img_path.split('/')[-1]
    results[0].save(filename=output_path)

print("Segmentation completed for all listed images.")
