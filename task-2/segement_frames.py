from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

num_frames = 24  

for i in range(1, num_frames + 1):
    frame_path = f'frames/frame_{i:04d}.jpg'  # input frames path
    results = model(frame_path)
    save_path = f'segmented_frames/seg_frame_{i:04d}.jpg'  # save segmented frames here
    results[0].save(filename=save_path)

print("Segmentation completed.")
