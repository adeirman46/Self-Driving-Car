import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.01
var_imu_w = 0.01
var_gnss  = 10.00
var_lidar = 1.00

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity

# UKF specific parameters
n_aug = 15  # Augmented state dimension
alpha = 0.1  # Spread of sigma points
kappa = 0  # Secondary scaling parameter
beta = 2  # Optimal for Gaussian distributions
lambda_ukf = alpha**2 * (n_aug + kappa) - n_aug  # Scaling parameter
n_sig = 2 * n_aug + 1  # Number of sigma points

# Weight calculations
weights_m = np.zeros(n_sig)
weights_c = np.zeros(n_sig)
weights_m[0] = lambda_ukf / (n_aug + lambda_ukf)
weights_c[0] = lambda_ukf / (n_aug + lambda_ukf) + (1 - alpha**2 + beta)
for i in range(1, n_sig):
    weights_m[i] = 1 / (2 * (n_aug + lambda_ukf))
    weights_c[i] = weights_m[i]

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our UKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 15, 15])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.eye(15)  # covariance of estimate
gnss_i  = 0
lidar_i = 0

#### 4. UKF Functions ##########################################################################

# def generate_sigma_points(mean, cov):
#     n = mean.shape[0]
#     sigma_points = np.zeros((2*n + 1, n))
#     sigma_points[0] = mean
    
#     L = np.linalg.cholesky((n + lambda_ukf) * cov)
    
#     for i in range(n):
#         sigma_points[i+1] = mean + L[i]
#         sigma_points[i+1+n] = mean - L[i]
    
#     return sigma_points

# def generate_sigma_points(mean, cov, lambda_ukf):
#     n = mean.shape[0]
#     sigma_points = np.zeros((2*n + 1, n))
#     sigma_points[0] = mean
    
#     # Adding a small value to the diagonal for numerical stability
#     jitter = 1e-6 * np.eye(n)
#     try:
#         L = np.linalg.cholesky((n + lambda_ukf) * cov + jitter)
#     except np.linalg.LinAlgError:
#         print("Cholesky decomposition failed, covariance matrix might not be positive definite.")
#         return None
    
#     for i in range(n):
#         sigma_points[i+1] = mean + L[i]
#         sigma_points[i+1+n] = mean - L[i]
    
#     return sigma_points

# def generate_sigma_points(mean, cov, lambda_ukf):
#     n = mean.shape[0]
#     sigma_points = np.zeros((2*n + 1, n))
#     sigma_points[0] = mean
    
#     # Adding a small value to the diagonal for numerical stability
#     jitter = 1e-6 * np.eye(n)
#     try:
#         L = np.linalg.cholesky((n + lambda_ukf) * cov + jitter)
#     except np.linalg.LinAlgError:
#         print("Cholesky decomposition failed, covariance matrix might not be positive definite.")
#         return None
    
#     for i in range(n):
#         sigma_points[i+1] = mean + L[i]
#         sigma_points[i+1+n] = mean - L[i]
    
#     return sigma_points
def generate_sigma_points(mean, cov, lambda_ukf):
    n = mean.shape[0]
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = mean
    
    # Adding a small value to the diagonal for numerical stability
    jitter = 1e-6 * np.eye(n)
    try:
        L = np.linalg.cholesky((n + lambda_ukf) * cov + jitter)
    except np.linalg.LinAlgError:
        print("Cholesky decomposition failed, covariance matrix might not be positive definite.")
        return None
    
    for i in range(n):
        sigma_points[i+1] = mean + L[i]
        sigma_points[i+1+n] = mean - L[i]
    
    return sigma_points

def unscented_transform(sigma_points, weights_m, weights_c):
    mean = np.sum(weights_m[:, None] * sigma_points, axis=0)
    
    cov = np.zeros((sigma_points.shape[1], sigma_points.shape[1]))
    for i in range(sigma_points.shape[0]):
        diff = sigma_points[i] - mean
        cov += weights_c[i] * np.outer(diff, diff)
    
    return mean, cov

def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # Generate sigma points
    x_aug = np.concatenate((p_check, v_check, Quaternion(*q_check).to_euler()))
    sigma_points = generate_sigma_points(x_aug, p_cov_check)
    
    # Predict measurement
    z_pred = sigma_points[:, :3]  # Only position is measured
    z_mean, S = unscented_transform(z_pred, weights_m, weights_c)
    
    # Add measurement noise
    S += np.eye(3) * sensor_var
    
    # Calculate cross correlation
    Tc = np.zeros((15, 3))
    for i in range(n_sig):
        x_diff = sigma_points[i] - x_aug
        z_diff = z_pred[i] - z_mean
        Tc += weights_c[i] * np.outer(x_diff, z_diff)
    
    # Kalman gain
    K = Tc @ np.linalg.inv(S)
    
    # Update state
    y_diff = y_k - z_mean
    x_update = x_aug + K @ y_diff
    
    # Update covariance
    p_cov_hat = p_cov_check - K @ S @ K.T
    
    return x_update[:3], x_update[3:6], Quaternion(euler=x_update[6:]).to_numpy(), p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
n_state = 15  # Position (3) + Velocity (3) + Quaternion (4) + Biases (6) (assumed for this case)

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    
    # 1. Generate sigma points
    x_aug = np.concatenate((p_est[k-1], v_est[k-1], Quaternion(*q_est[k-1]).to_euler(), np.zeros(6)))  # Assuming 6 bias states
    
    # Debugging statement
    print(f"Iteration {k}, x_aug shape: {x_aug.shape}, p_cov[k-1] shape: {p_cov[k-1].shape}")

    # Ensure the covariance matrix is the correct shape
    if p_cov[k-1].shape != (n_state, n_state):
        print(f"Error: covariance matrix shape is {p_cov[k-1].shape}, expected ({n_state}, {n_state})")
        break

    sigma_points = generate_sigma_points(x_aug, p_cov[k-1], lambda_ukf)
    if sigma_points is None:
        # Handle the failure case (e.g., skip update, use previous state, etc.)
        p_est[k] = p_est[k-1]
        v_est[k] = v_est[k-1]
        q_est[k] = q_est[k-1]
        p_cov[k] = p_cov[k-1]
        continue
    
    # 2. Predict sigma points
    for i in range(n_sig):
        # Extract states
        p = sigma_points[i, :3]
        v = sigma_points[i, 3:6]
        q = Quaternion(euler=sigma_points[i, 6:9])
        biases = sigma_points[i, 9:]  # Extract biases (if included)
        
        # IMU motion model
        p = p + v * delta_t + 0.5 * delta_t**2 * (q.to_mat() @ imu_f.data[k-1] + g)
        v = v + delta_t * (q.to_mat() @ imu_f.data[k-1] + g)
        q = Quaternion(axis_angle=imu_w.data[k-1] * delta_t).quat_mult_right(q)
        
        # Store predicted sigma points
        sigma_points[i, :3] = p
        sigma_points[i, 3:6] = v
        sigma_points[i, 6:9] = q.to_euler()
        sigma_points[i, 9:] = biases  # Store biases (if included)
    
    # 3. Predict mean and covariance
    x_pred, p_cov[k] = unscented_transform(sigma_points, weights_m, weights_c)
    
    # 4. Update state estimates
    p_est[k] = x_pred[:3]
    v_est[k] = x_pred[3:6]
    q_est[k] = Quaternion(euler=x_pred[6:9]).to_numpy()
    
    # 5. Measurement updates
    if gnss_i < gnss.data.shape[0] and gnss.t[gnss_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i], p_est[k], v_est[k], q_est[k])
        gnss_i += 1
    
    if lidar_i < lidar.data.shape[0] and lidar.t[lidar_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i], p_est[k], v_est[k], q_est[k])
        lidar_i += 1


#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# # Pt. 1 submission
# p1_indices = [9000, 9400, 9800, 10200, 10600]
# p1_str = ''
# for val in p1_indices:
#     for i in range(3):
#         p1_str += '%.3f ' % (p_est[val, i])
# with open('pt1_submission.txt', 'w') as file:
#     file.write(p1_str)

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)

# Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]
p3_str = ''
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])
with open('pt3_submission.txt', 'w') as file:
    file.write(p3_str)