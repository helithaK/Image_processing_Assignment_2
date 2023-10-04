import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt
import cv2 as cv

# List to store selected points
selected_points = []

# Callback function for mouse click events
def handle_click(event, x, y, flags, param):
    global selected_points
    if event == cv.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv.circle(input_img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow('Input Image', input_img)

        if len(selected_points) == 4:
            cv.destroyAllWindows()

# Load the input image
input_img = cv.cvtColor(cv.imread('input_image.jpg'), cv.COLOR_BGR2RGB)

# Create a window to display the input image and set mouse callback
cv.imshow('Input Image', input_img)
cv.setMouseCallback('Input Image', handle_click)
cv.waitKey(0)

# Print selected points
print("Selected Points:")
for point in selected_points:
    print(point)

# Load the output image and flag image
output_img = cv.imread('output_image.jpg')
flag_img = cv.imread('flag.png')

# Convert selected points to NumPy arrays
points_input = np.array(selected_points, dtype=np.float32)
points_flag = np.array([[0, 0], [flag_img.shape[1], 0], [flag_img.shape[1], flag_img.shape[0]], [0, flag_img.shape[0]]], dtype=np.float32)

# Find the homography matrix to transform the flag to match the selected points
homography_matrix, _ = cv.findHomography(points_flag, points_input)

# Warp the flag image
flag_warped = cv.warpPerspective(flag_img, homography_matrix, (input_img.shape[1], input_img.shape[0]))

# Convert the warped flag image to RGB format
flag_warped_rgb = cv.cvtColor(flag_warped, cv.COLOR_BGR2RGB)

# Set alpha for blending
alpha = 0.6

# Create a composite image
composite_img = cv.addWeighted(input_img, 1, flag_warped_rgb, alpha, 0)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(input_img)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(flag_warped_rgb)
plt.title('Warped Flag')
plt.subplot(1, 3, 3)
plt.imshow(composite_img)
plt.title('Composite Image')
plt.tight_layout()
plt.show()
