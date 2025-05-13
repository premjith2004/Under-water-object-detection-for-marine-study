# -*- coding: utf-8 -*-
"""underwater object detection
"""

//pip install opencv-python numpy

import cv2
import numpy as np

# Load the video capture device (e.g., a webcam or a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range of colors to detect (e.g., blue or green for underwater objects)
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([130, 255, 255])
    
    # Threshold the HSV image to get only the desired colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Find contours of the detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around the detected objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow('Underwater Object Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
