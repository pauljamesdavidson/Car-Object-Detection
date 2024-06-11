import cv2
from ultralytics import YOLO
import torch
import os

# Initialize a new YOLO object with the desired configuration
model = YOLO()

# Load the state dictionary, adjusting for the size mismatch error
state_dict = torch.load('runs/detect/train3/weights/best.pt')
model_state_dict = model.state_dict()

# Filter out unnecessary keys from the state dictionary
state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

# Update the model's state dictionary with the filtered state dictionary
model_state_dict.update(state_dict)

# Load the updated state dictionary into the model
model.load_state_dict(model_state_dict)

# Open the input video
input_video_path = 'videos/computer vision.mp4'
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file {input_video_path}")
    exit()

# Get the width and height of the frames in the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_video_path = 'predicted_video/output_video.mp4'
output_dir = os.path.dirname(output_video_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Predict objects in the frame
    results = model.predict(frame)

    # Inspect the structure of results
    if frame_count == 1:
        print(f"Results structure: {type(results)}, {len(results)} elements")
        print(f"First element type: {type(results[0])}")

    # Accessing the result data properly
    for result in results[0].boxes:  # Adjusting to access the 'boxes' attribute
        box = result.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates
        confidence = result.conf[0].cpu().numpy()  # Confidence score
        class_id = int(result.cls[0].cpu().numpy())  # Class ID

        # Get class name from ID using model.names
        label = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Put label text
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames")

# Release the video objects
cap.release()
out.release()

print(f'Video saved as {output_video_path}')
