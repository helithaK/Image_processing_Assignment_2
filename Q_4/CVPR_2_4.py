import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load images
image1, image5 = cv.imread("pic1.jpg"), cv.imread("pic5.jpg")

# Create a SIFT object
sift_detector = cv.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift_detector.detectAndCompute(image1, None)
keypoints5, descriptors5 = sift_detector.detectAndCompute(image5, None)

# Create a Brute-Force Matcher for feature matching
bf_matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

# Match descriptors between the two images
matches = sorted(bf_matcher.match(descriptors1, descriptors5), key=lambda x: x.distance)

# Draw matching keypoints between the images
matched_image = cv.drawMatches(image1, keypoints1, image5, keypoints5, matches[:250], image5, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image
fig, ax = plt.subplots(figsize=(7, 7))
matched_image = cv.cvtColor(matched_image, cv.COLOR_BGR2RGB)
ax.set_title("SIFT Feature Matching")
ax.imshow(matched_image)
plt.show()

# Define a list of images
images = [cv.imread("pic1.jpg"), cv.imread("pic2.jpg"), cv.imread("pic3.jpg"), cv.imread("pic4.jpg"), cv.imread("pic5.jpg")]

# Convert images to grayscale
gray_images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in images]

# Define a function to generate random numbers with replacement
def random_indices(n, t):
    return np.random.randint(n, size=t)

# Define a function to compute homography matrix
def compute_homography(source_points, target_points):
    # Implementation of Homography calculation...
    pass

# Parameters for RANSAC
confidence = 0.99
sample_size = 4
threshold = 0.5
num_iterations = int(np.ceil(np.log(1 - confidence) / np.log(1 - ((1 - threshold) ** sample_size))))

# List to store computed homography matrices
homography_matrices = []

# Loop through images for feature matching and homography calculation
for i in range(4):
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift_detector.detectAndCompute(gray_images[i], None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(gray_images[i + 1], None)

    # Match descriptors using Brute-Force Matcher
    matches = sorted(bf_matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)

    # Extract source and target points
    source_points = [keypoints1[match.queryIdx].pt for match in matches]
    target_points = [keypoints2[match.trainIdx].pt for match in matches]

    # RANSAC-based homography estimation
    best_inliers = 0
    best_homography = None

    for _ in range(num_iterations):
        # Randomly select sample_size points
        random_indices = random_indices(len(source_points), sample_size)
        sample_source = [source_points[idx] for idx in random_indices]
        sample_target = [target_points[idx] for idx in random_indices]

        # Compute homography matrix using the sample points
        current_homography = compute_homography(sample_source, sample_target)

        # Count inliers using the current homography
        inliers = 0
        for idx, src_point in enumerate(source_points):
            # Transform source point using the current homography
            transformed_point = np.dot(current_homography, [src_point[0], src_point[1], 1])
            transformed_point /= transformed_point[2]

            # Calculate Euclidean distance between transformed point and target point
            euclidean_distance = np.sqrt((transformed_point[0] - target_points[idx][0])**2 + (transformed_point[1] - target_points[idx][1])**2)

            # Check if the point is an inlier
            if euclidean_distance < threshold:
                inliers += 1

        # Update best homography if the current one has more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_homography = current_homography

    # Add the best homography to the list
    homography_matrices.append(best_homography)

# Compute the cumulative homography matrix
final_homography = np.linalg.multi_dot(homography_matrices)

# Print the computed homography and the provided homography
print("Computed Homography Matrix:\n", final_homography)
print("Provided Homography Matrix:\n", open("provided_homography.txt", 'r').read())

# Warp the first image using the computed homography
transformed_image1 = cv.warpPerspective(images[0], final_homography, (np.shape(images[4])[1], np.shape(images[4])[0]))

# Create the final stitched image
final_stitched_image = cv.add(images[4], transformed_image1)

# Display the original images and the stitched image
fig, ax = plt.subplots(1, 4, figsize=(15, 15))
ax[0].imshow(cv.cvtColor(images[0], cv.COLOR_BGR2RGB))
ax[0].set_title("Image 1")
ax[1].imshow(cv.cvtColor(images[4], cv.COLOR_BGR2RGB))
ax[1].set_title("Image 5")
ax[2].imshow(cv.cvtColor(transformed_image1, cv.COLOR_BGR2RGB))
ax[2].set_title("Transformed Image")
ax[3].imshow(cv.cvtColor(final_stitched_image, cv.COLOR_BGR2RGB))
ax[3].set_title("Stitched Image")
plt.show()
