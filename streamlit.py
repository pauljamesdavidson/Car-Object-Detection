import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import torch
import os
import time

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

# Define the detect_video function
def detect_video(video_file):
    with st.spinner('Processing...'):
        # Open the input video
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            st.error(f"Error: Unable to open video file {video_file}")
            return

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the width and height of the frames in the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for H.264 video
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

        start_time = time.time()
        processed_frames = 0

        estimated_time_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict objects in the frame
            results = model.predict(frame)

            # Accessing the result data properly
            for result in results[0].boxes:  # Adjusting to access the 'boxes' attribute
                box = result.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates
                confidence = result.conf[0].cpu().numpy()  # Confidence score
                class_id = int(result.cls[0].cpu().numpy())  # Class ID

                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Put label text
                label_text = f"{model.names[class_id]} {confidence:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the output video
            out.write(frame)

            processed_frames += 1

            # Calculate estimated time remaining
            elapsed_time = time.time() - start_time
            estimated_time_remaining = (total_frames - processed_frames) * (elapsed_time / processed_frames)

            estimated_time_placeholder.text(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")

        # Release the video objects
        cap.release()
        out.release()

        st.success(f'Video processed and saved as {output_video_path}')

        # Display processed video using OpenCV
        st.write("**Processed Video**")
        st.video(output_video_path, format='mp4')


# Streamlit App
st.title("Car Object Detection")

# Sidebar
st.sidebar.title("Upload and Detect")
uploaded_file = st.sidebar.file_uploader("Upload a Car Video", type=["mp4"])

# Main page
if uploaded_file:
    st.sidebar.write("**Uploaded Video**")
    st.sidebar.video(uploaded_file, format='mp4')

    if st.button("Detect"):
        # Save the uploaded file to a temporary location
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
        detect_video(temp_video_path)
else:
    st.write("**Video on Main Page**")
    main_page_video = 'videos/computer vision.mp4'
    st.video(main_page_video, format='mp4')
    if st.button("Detect Video on Main Page"):
        detect_video(main_page_video)
