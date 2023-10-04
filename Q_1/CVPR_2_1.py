import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import spatial

# Load the image
image_path = "images/the_berry_farms_sunflower_field.jpeg"
img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check if the image was loaded successfully
assert img_gray is not None

# Display the original image
fig, ax = plt.subplots()
ax.imshow(img_gray, interpolation='nearest', cmap="gray")
ax.set_axis_off()
ax.plot()
plt.show()

# Print the shape of the image
print("Image shape:", img_gray.shape)
# Define the LoG (Laplacian of Gaussian) filter
def LoG(sigma):
    # Calculate the filter size based on sigma
    filter_size = int(np.ceil(sigma * 6))
    # Create 1D Gaussian filters along x and y axes
    y, x = np.ogrid[-filter_size // 2:filter_size // 2 + 1, -filter_size // 2:filter_size // 2 + 1]
    y_filter = np.exp(-(y * y / (2.0 * sigma * sigma)))
    x_filter = np.exp(-(x * x / (2.0 * sigma * sigma)))
    # Create the LoG filter
    LoG_filter = (-(2 * sigma ** 2) + (x * x + y * y)) * (x_filter * y_filter) * (1 / (2 * np.pi * sigma ** 4))
    return LoG_filter
# Convolve the image with the LoG filter at multiple scales
def LoG_convolve(img_gray):
    log_images = []
    k = 1.414
    sigma = 3.0
    img_gray = img_gray / 255.0
    for i in range(1, 10):
        sigma_i = sigma * k ** i
        filter_log = LoG(sigma_i)
        image = cv2.filter2D(img_gray, -1, filter_log)
        image = np.pad(image, ((1, 1), (1, 1)), 'constant')
        image = np.square(image)
        log_images.append(image)

    log_image_np = np.array(log_images)
    return log_image_np


# Perform LoG convolution
log_image_np = LoG_convolve(img_gray)
print("LoG Convolution result shape:", log_image_np.shape)


# Calculate the overlap between two blobs
def blob_overlap(blob1, blob2):
    n_dim = len(blob1) - 1
    root_ndim = math.sqrt(n_dim)
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim
    d = math.sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))

    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return 1
    else:
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d)))
        return area / (math.pi * (min(r1, r2) ** 2))


# Remove redundant blobs
def redundancy(blobs_array, overlap_threshold):
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))

    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blob_overlap(blob1, blob2) > overlap_threshold:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    return np.array([b for b in blobs_array if b[-1] > 0])


# Detect blobs in the LoG-convolved image
def detect_blobs(log_image_np):
    coordinates = []
    (h, w) = img_gray.shape
    k = 1.414

    for i in range(1, h):
        for j in range(1, w):
            slice_img = log_image_np[:, i - 1:i + 2, j - 1:j + 2]
            result = np.amax(slice_img)

            if result >= 0.03:
                z, x, y = np.unravel_index(slice_img.argmax(), slice_img.shape)
                coordinates.append((i + x - 1, j + y - 1, k ** z * sigma))

    return coordinates


# Detect blobs and remove redundancy
coordinates = list(set(detect_blobs(log_image_np)))
coordinates = np.array(coordinates)
coordinates = redundancy(coordinates, 0.5)

# Display the image with detected blobs
fig, ax = plt.subplots()
nh, nw = img_gray.shape

ax.imshow(img_gray, interpolation='nearest', cmap="gray")
radius = 0

for blob in coordinates:
    y, x, r = blob
    c = plt.Circle((x, y), r * 1.414, color='red', linewidth=1, fill=False)

    if r > radius:
        radius = r * 1.414
        (param_x, param_y) = (x, y)

    ax.add_patch(c)

ax.plot()
ax.set_axis_off()
plt.show()

# Print the parameters of the detected blobs
print("Detected Blobs (y, x, radius):")
for blob in coordinates:
    print(blob)
