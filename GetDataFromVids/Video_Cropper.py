import cv2

# Load the video
input_video = cv2.VideoCapture('BottomRightPS.mp4')

# Define the crop region
top, right, bottom, left = 977, 2535, 1263, 435

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' codec for MP4 format
output_movie = cv2.VideoWriter('PS_Crop.mp4', fourcc, 30, (right-left, bottom-top))

while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Crop the frame
    crop_img = frame[top:bottom, left:right]

    # Write the cropped frame to the output video
    output_movie.write(crop_img)

# Release the video file and close the video writer
input_video.release()
output_movie.release()
