import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('PVC_Crop.mp4')

# Initialize variables
prev_frame = None
threshold = 1 # Adjust this value based on your video
frame_counter = 1 # Initialize frame counter

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frames to grayscale for easier comparison
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(gray_frame, prev_frame)
        if np.mean(diff) > threshold:
            # Save the screenshot with the frame counter
            cv2.imwrite(f'PVC_Frame_{frame_counter}.png', frame)
            print(f"Graph changed, frame_{frame_counter}.png saved.")
            frame_counter += 1 # Increment the frame counter

    # Update previous frame
    prev_frame = gray_frame

# Release the video capture
cap.release()
