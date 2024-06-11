import os
import cv2

# Directory containing the images
image_dir = 'data/testing_images'

# Check if the directory exists
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"The directory {image_dir} does not exist.")

# Get sorted list of image filenames
image_files = sorted(os.listdir(image_dir))

# Check if there are any images in the directory
if not image_files:
    raise FileNotFoundError(f"No images found in the directory {image_dir}.")

# Define the codec and create VideoWriter object
output_dir = 'videos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
fps = 10  # Frames per second
frame_size = (1280, 720)  # Change this to your frame size (width, height)

# Initialize VideoWriter
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Iterate over each image file and write to video
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if frame is None:
        print(f"Warning: Unable to load image {image_path}. Skipping...")
        continue

    # Resize the frame to match the frame size
    frame = cv2.resize(frame, frame_size)
    
    # Write the frame to the video
    out.write(frame)

# Release the VideoWriter
out.release()

print(f'Video saved as {output_file}')
