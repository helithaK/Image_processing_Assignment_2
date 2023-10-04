import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tikzplotlib

# Constants
num_points = 100
half_num_points = num_points // 2
circle_radius = 10
circle_center_x, circle_center_y = 2, 3
circle_noise_scale = circle_radius / 16

# Generate random points for the circle
circle_angles = np.random.uniform(0, 2 * np.pi, half_num_points)
circle_noise = circle_noise_scale * np.random.randn(half_num_points)
circle_x = circle_center_x + (circle_radius + circle_noise) * np.cos(circle_angles)
circle_y = circle_center_y + (circle_radius + circle_noise) * np.sin(circle_angles)
circle_points = np.hstack((circle_x.reshape(half_num_points, 1), circle_y.reshape(half_num_points, 1)))

# Generate random points for the line
line_noise_scale = 1.0
line_slope, line_intercept = -1, 2
line_x = np.linspace(-12, 12, half_num_points)
line_y = line_slope * line_x + line_intercept + line_noise_scale * np.random.randn(half_num_points)
line_points = np.hstack((line_x.reshape(half_num_points, 1), line_y.reshape(half_num_points, 1)))

# Combine all points
data_points = np.vstack((circle_points, line_points))

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Scatter plot for the line points
ax.scatter(line_points[:, 0], line_points[:, 1], label='Line')

# Scatter plot for the circle points
ax.scatter(circle_points[:, 0], circle_points[:, 1], label='Circle')

# Ground truth circle
ground_truth_circle = plt.Circle((circle_center_x, circle_center_y), circle_radius, color='g', fill=False, label='Ground truth circle')
ax.add_patch(ground_truth_circle)
ax.plot(circle_center_x, circle_center_y, '+', color='g')

ground_truth_line_x = np.array([min(ax.get_xlim()), max(ax.get_xlim())])
ground_truth_line_y = line_slope * ground_truth_line_x + line_intercept
plt.plot(ground_truth_line_x, ground_truth_line_y, color='m', label='Ground truth line')

plt.legend()
plt.show()

# Line estimation using RANSAC
import math

# Constants
num_data_points = data_points.shape[0]
dataset = data_points

def line_equation_from_points(x1, y1, x2, y2):
    """ Return the line equation in the form ax + by = d"""
    delta_x = x2 - x1
    delta_y = y2 - y1
    magnitude = math.sqrt(delta_x ** 2 + delta_y ** 2)
    a = delta_y / magnitude
    b = -delta_x / magnitude
    d = (a * x1) + (b * y1)
    return a, b, d

def total_least_squares_error(x, indices):
    """ Return the total least squares error for the line model"""
    a, b, d = x[0], x[1], x[2]
    return np.sum(np.square(a * dataset[indices, 0] + b * dataset[indices, 1] - d))

def constraint_function(x):
    """ Constraint function to ensure unit circle"""
    return x[0] ** 2 + x[1] ** 2 - 1

constraints = ({'type': 'eq', 'fun': constraint_function})

def consensus_line(X, x, t):
    """ Compute the inliers """
    a, b, d = x[0], x[1], x[2]
    error = np.abs(a * dataset[:, 0] + b * dataset[:, 1] - d)
    return error < t

threshold = 1.0
min_inliers = 0.4 * num_data_points
min_sample_points = 2
max_iterations = 50
iteration = 0
best_model_line = []
best_error = np.inf
best_sample_line = []
best_inliers_line = []

while iteration < max_iterations:
    sample_indices = np.random.randint(0, num_data_points, min_sample_points)
    initial_estimate = np.array([1, 1, 0])
    result = minimize(fun=total_least_squares_error, args=sample_indices, x0=initial_estimate, tol=1e-6, constraints=constraints, options={'disp': True})
    inliers = consensus_line(dataset, result.x, threshold)

    if np.sum(inliers) > min_inliers:
        initial_estimate = result.x
        result = minimize(fun=total_least_squares_error, args=inliers, x0=initial_estimate, tol=1e-6, constraints=constraints, options={'disp': True})

        if result.fun < best_error:
            best_model_line = result.x
            best_error = result.fun
            best_sample_line = dataset[sample_indices, :]
            best_inliers_line = inliers

    iteration += 1

print('Best line model', best_model_line)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(dataset[:, 0], dataset[:, 1], label='All points')
ax.scatter(dataset[best_inliers_line, 0], dataset[best_inliers_line, 1], color='y', label='Line Inliers')
ax.scatter(best_sample_line[:, 0], best_sample_line[:, 1], color='r', label='Best samples for line')
x_min, x_max = ax.get_xlim()
x_ = np.array([x_min, x_max])
y_ = (-best_model_line[1] * x_ + best_model_line[2]) / best_model_line[1]
plt.plot(x_, y_, label='RANSAC line')
x_ = np.array([x_min, x_max])
y_ = line_slope * x_ + line_intercept
plt.plot(x_, y_, color='m', label='Ground truth line')

# Plotting the inliers
print("Ratio of inliers =", len(best_inliers_line) / num_data_points * 100, "%")
ax.scatter(best_inliers_line[:, 0], best_inliers_line[:, 1], color='c', label='Circle Inliers')
ax.scatter(best_sample_line[:, 0], best_sample_line[:, 1], color='red', label='Best fitting samples')

plt.legend()

# Function to calculate the circle equation from three points
def circle_equation_from_points(points):
    p1, p2, p3 = points[0], points[1], points[2]
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return (cx, cy), radius

# Function to get inliers based on a given circle
def get_circle_inliers(data_points, center, radius):
    inliers = []
    threshold = radius / 5

    for i in range(len(data_points)):
        error = np.sqrt((data_points[i][0] - center[0]) ** 2 + (data_points[i][1] - center[1]) ** 2) - radius
        if error < threshold:
            inliers.append(data_points[i])

    return np.array(inliers)

# Function to randomly sample 3 points from the data
def random_sample(data_points):
    sample_list = []
    random.seed(0)
    random_indices = random.sample(range(1, len(data_points)), 3)
    for i in random_indices:
        sample_list.append(data_points[i])
    return np.array(sample_list)

# Function to calculate the center and radius of a circle using the least squares method
def estimate_circle(x_m, y_m, points):
    x_ = points[:, 0]
    y_ = points[:, 1]
    center_estimate = x_m, y_m
    center, _ = optimize.leastsq(f_2, center_estimate, (x_, y_))
    xc, yc = center
    Ri = calc_R(x_, y_, *center)
    R = Ri.mean()
    return (xc, yc), R

# Function to apply RANSAC to fit a circle to a set of points
def ransac_circle(data_points, iterations):
    best_sample = []
    best_center_sample = (0, 0)
    best_radius_sample = 0
    best_inliers = []
    max_inliers = 20

    for i in range(iterations):
        samples = random_sample(data_points)
        center, radius = circle_equation_from_points(samples)
        inliers = get_circle_inliers(data_points, center, radius)
        num_inliers = len(inliers)

        if num_inliers > max_inliers:
            best_sample = samples
            max_inliers = num_inliers
            best_center_sample = center
            best_radius_sample = radius
            best_inliers = inliers

    print("Center of Sample =", best_center_sample)
    print("Radius of Sample =", best_radius_sample)

    return best_center_sample, best_radius_sample, best_sample, best_inliers

# Calculate the RANSAC outputs for the data set
center_circle, radius_circle, sample_circle, inliers_circle = ransac_circle(data_points, 50000)
print("Sample:", sample_circle)
print("Data Points Shape:", data_points.shape)
print("Inliers:", inliers_circle)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
plt.scatter(data_points[:, 0], data_points[:, 1], color='blue', label='Data points')
circle = plt.Circle(center_circle, radius_circle, fill=False, label='Circle through the best fitting samples', color='k')
ax.add_patch(circle)

print("Ratio of inliers =", len(inliers_circle) / half_num_points * 100, "%")
ax.scatter(inliers_circle[:, 0], inliers_circle[:, 1], color='green', label='Inliers')

ransac_center_circle, ransac_radius_circle = estimate_circle(center_circle[0], center_circle[1], inliers_circle)
print("Center of RANSAC Circle =", ransac_center_circle)
print("Radius of RANSAC Circle =", ransac_radius_circle)

circle = plt.Circle(ransac_center_circle, ransac_radius_circle, fill=False, label='RANSAC output circle', color='b')
ax.add_patch(circle)
ax.scatter(sample_circle[:, 0], sample_circle[:, 1], color='red', label='Best fitting samples')
ax.legend()

# Plotting the RANSAC outputs for the data set
dataset = data_points
center_circle, radius_circle, sample_circle, inliers_circle = ransac_circle(dataset, 50000)
print("Sample:", sample_circle)
print("Data Points Shape:", dataset.shape)
print("Inliers:", inliers_circle)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
plt.scatter(data_points[:, 0], data_points[:, 1], color='blue', label='Data points')
circle = plt.Circle(center_circle, radius_circle, fill=False, label='Circle through the best fitting samples', color='k')
ax.add_patch(circle)

print("Ratio of inliers =", len(inliers_circle) / half_num_points * 100, "%")
ax.scatter(inliers_circle[:, 0], inliers_circle[:, 1], color='green', label='Inliers')

ransac_center_circle, ransac_radius_circle = estimate_circle(center_circle[0], center_circle[1], inliers_circle)
print("Center of RANSAC Circle =", ransac_center_circle)
print("Radius of RANSAC Circle =", ransac_radius_circle)

circle = plt.Circle(ransac_center_circle, ransac_radius_circle, fill=False, label='RANSAC circle', color='b')
ax.add_patch(circle)
ax.scatter(sample_circle[:, 0], sample_circle[:, 1], color='red', label='Best fitting samples')
ax.legend()
