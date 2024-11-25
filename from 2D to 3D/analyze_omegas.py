
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt
from extract_flight_data import FlightAnalysis
from visualize import Visualizer
import plotly.graph_objects as go
from visualize import create_gif_one_movie
# matplotlib.use('TkAgg')


def check_high_blind_axis_omegas(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["body_wx_take", "body_wy_take", "body_wz_take"])
    # Extract omega values for Mahalanobis filtering
    omega, _, _, _ = get_3D_attribute_from_df(df)

    # Compute the mean and covariance matrix of omega
    mean = np.mean(omega, axis=0)
    covariance_matrix = np.cov(omega, rowvar=False)

    # Filter omega rows based on Mahalanobis distance
    filtered_indices = []
    for i, row in enumerate(omega):
        dist = mahalanobis(row, mean, np.linalg.inv(covariance_matrix))
        if dist < 3:
            filtered_indices.append(i)

    # Filter the dataframe based on Mahalanobis distance
    filtered_df = df.iloc[filtered_indices]
    filtered_omega, _, _, _ = get_3D_attribute_from_df(filtered_df)
    vec_all, yaw_all, pitch_all, yaw_std_all, pitch_std_all = get_pca_points(filtered_omega)

    # Project omega vectors onto vec_all and calculate the distance from the origin
    projections = np.dot(filtered_omega, vec_all)
    distances = np.abs(projections)

    # Define a threshold for filtering based on distances (for example, top 20% based on distance)
    top = 20
    threshold_distance = np.percentile(distances, 100 - top)

    # Filter rows where the projection distance is within the threshold
    final_filtered_indices = [i for i, distance in enumerate(distances) if distance >= threshold_distance]
    all_indices_filtered_df = np.arange(len(distances))
    remaining_indices = list(set(all_indices_filtered_df) - set(final_filtered_indices))

    # Get the corresponding original indices
    final_filtered_df = filtered_df.iloc[final_filtered_indices]
    non_filtered_df = filtered_df.iloc[remaining_indices]

    high_omegas, _, _, _ = get_3D_attribute_from_df(final_filtered_df)
    rest_of_omegas , _, _, _ = get_3D_attribute_from_df(non_filtered_df)
    high_omegas_torques, _, _, _ =  get_3D_attribute_from_df(final_filtered_df, attirbutes=["torque_body_x_take",
                                                                               "torque_body_y_take",
                                                                               "torque_body_z_take"])
    rest_of_torques, _, _, _ = get_3D_attribute_from_df(non_filtered_df, attirbutes=["torque_body_x_take",
                                                                               "torque_body_y_take",
                                                                               "torque_body_z_take"])

    dir = os.path.dirname(csv_file)
    output_file_path = os.path.join(dir, 'high 20 percent omegas.csv')
    final_filtered_df.to_csv(output_file_path, index=False)

    # Plotting
    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(high_omegas[:, 0], high_omegas[:, 1], high_omegas[:, 2], s=1, color='blue', label='top 20')
    # ax.scatter(rest_of_omegas[:, 0], rest_of_omegas[:, 1], rest_of_omegas[:, 2], s=1, color='red', label='rest')
    ax.scatter(rest_of_torques[:, 0], rest_of_torques[:, 1], rest_of_torques[:, 2], s=5, color='red' , label='rest')
    ax.set_aspect('equal')
    plt.legend()
    plt.show()


def create_rotating_frames(N, dt, omegas):
    x_body = np.array([1, 0, 0])
    y_body = np.array([0, 1, 0])
    z_body = np.array([0, 0, 1])

    # Create arrays to store the frames
    x_frames = np.zeros((N, 3))
    y_frames = np.zeros((N, 3))
    z_frames = np.zeros((N, 3))

    # Initialize the frames with the initial reference frame
    x_frames[0] = x_body
    y_frames[0] = y_body
    z_frames[0] = z_body

    for i in range(1, N):
        omega = omegas[i]
        # Construct the skew-symmetric matrix for omega
        omega_matrix = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])

        # Apply the angular velocity tensor to generate the frames
        R = np.stack((x_frames[i - 1], y_frames[i - 1], z_frames[i - 1]), axis=-1)
        dR = np.dot(omega_matrix, R) * dt
        R_new = R + dR

        # Ensure orthogonality and normalization
        U, _, Vt = np.linalg.svd(R_new, full_matrices=False)
        R_new = np.dot(U, Vt)

        x_frames[i] = R_new[:, 0]
        y_frames[i] = R_new[:, 1]
        z_frames[i] = R_new[:, 2]

    return x_frames, y_frames, z_frames


def experiment():
    # Initial reference frame
    N = 10000
    dt = 1
    t = np.linspace(0, 2 * np.pi, N)  # Generate time array
    # Generate N omegas with sine function
    omegas = np.vstack([
        0.0001 * np.sin(t),
        0.0002 * np.sin(2 * t),
        0.0001 * np.sin(3 * t)
    ]).T

    x_frames, y_frames, z_frames = create_rotating_frames(N, dt, omegas)

    omega_lab, omega_body, angular_speed_lab, angular_speed_body = FlightAnalysis.get_angular_velocities(
        x_frames, y_frames, z_frames, start_frame=0, end_frame=N, sampling_rate=1)
    omega_lab = np.radians(omega_lab)

    percentage_error = np.abs((omegas - omega_lab) / (omegas + 0.00000001)) * 100
    mean_percentage_error = percentage_error.mean(axis=0)

    plt.plot(omegas, color='r')
    plt.plot(omega_lab, color='b')
    plt.show()
    return


def create_rotating_frames_yaw_pitch_roll(N, yaw_angles, pitch_angles, roll_angles):
    x_body = np.array([1, 0, 0])
    y_body = np.array([0, 1, 0])
    z_body = np.array([0, 0, 1])

    # Create arrays to store the frames
    x_frames = np.zeros((N, 3))
    y_frames = np.zeros((N, 3))
    z_frames = np.zeros((N, 3))

    # Initialize the frames with the initial reference frame
    x_frames[0] = x_body
    y_frames[0] = y_body
    z_frames[0] = z_body

    for i in range(0, N):
        # Get the yaw, pitch, and roll angles for the current frame
        yaw_angle = yaw_angles[i]
        pitch_angle = pitch_angles[i]
        roll_angle = roll_angles[i]

        R = FlightAnalysis.euler_rotation_matrix(yaw_angle, pitch_angle, roll_angle).T

        # Apply the rotation to the initial body frame
        x_frames[i] = R @ x_body
        y_frames[i] = R @ y_body
        z_frames[i] = R @ z_body

    return x_frames, y_frames, z_frames


# Example usage
def experiment_2(what_to_enter):
    # what_to_enter: coule be either omega or yaw, pitch roll
    N = 1000
    if what_to_enter == 'omega':
        dt = 1
        # Generate N omegas with sine function
        d = 0.001
        wx = np.zeros(N)
        wy = np.zeros(N)
        wz = np.zeros(N)

        # wx = 3 * d * np.linspace(0, 10, N)
        wy = 2 * d * np.ones(N)
        wz = 10 * d * np.ones(N)
        omegas = np.vstack([
            wx,
            wy,
            wz
        ]).T

        x_frames, y_frames, z_frames = create_rotating_frames(N=N, omegas=omegas, dt=1)
        Rs = np.stack((x_frames, y_frames, z_frames), axis=-1)

        # rs = [scipy.spatial.transform.Rotation.from_matrix(Rs[i]) for i in range(N)]
        # yaw = np.array([rs[i].as_euler('zyx', degrees=False)[0] for i in range(N)])
        # pitch = np.array([rs[i].as_euler('zyx', degrees=False)[1] for i in range(N)])
        # roll = np.array([rs[i].as_euler('zyx', degrees=False)[2] for i in range(N)])

        yaw = np.unwrap(np.array([np.arctan2(r[1, 0], r[0, 0]) for r in Rs]))
        pitch = -np.unwrap(np.array([np.arcsin(-r[2, 0]) for r in Rs]), period=np.pi)
        roll = np.unwrap(np.array([np.arctan2(r[2, 1], r[2, 2]) for r in Rs]))

        yaw_mine = np.radians(FlightAnalysis.get_body_yaw(x_frames))
        pitch_mine = np.radians(FlightAnalysis.get_body_pitch(x_frames))
        roll_mine = np.radians(FlightAnalysis.get_body_roll(phi=np.degrees(yaw_mine),
                                                            theta=np.degrees(pitch_mine),
                                                            x_body=x_frames,
                                                            y_body=y_frames,
                                                            yaw=np.degrees(yaw_mine),
                                                            pitch=np.degrees(pitch_mine),
                                                            start=0,
                                                            end=N, ))

        is_close = (np.all(np.isclose(yaw_mine, yaw))
                    and np.all(np.isclose(pitch_mine, pitch))
                    and np.all(np.isclose(roll_mine, roll)))
        print(f"is mine like the other way? {is_close}")

        plt.title("yaw, pitch, roll")
        plt.plot(yaw, label='yaw', c='r')
        plt.plot(pitch, label='pitch', c='g')
        plt.plot(roll, label='roll', c='b')
        plt.plot(yaw_mine, label='yaw mine', c='r', linestyle='--')
        plt.plot(pitch_mine, label='pitch mine', c='g', linestyle='--')
        plt.plot(roll_mine, label='roll mine', c='b', linestyle='--')
        plt.legend()
        plt.show()
        pass
    else:
        yaw = np.zeros(N)
        pitch = np.zeros(N)
        roll = np.zeros(N)

        yaw = np.linspace(0, 2 * 2 * np.pi, N)  # Example yaw angles for each frame
        pitch = np.linspace(0,2 * 2*np.pi, N)  # Example pitch angles for each frame
        roll = np.linspace(0, 2 * 2 * np.pi, N)  # Example roll angles for each frame
        x_frames, y_frames, z_frames = create_rotating_frames_yaw_pitch_roll(N, yaw, pitch, roll)

    yaw_dot = np.gradient(yaw)
    roll_dot = np.gradient(roll)
    pitch_dot = np.gradient(pitch)

    p, q, r = FlightAnalysis.get_pqr_calculation(-np.degrees(pitch), -np.degrees(pitch_dot), np.degrees(roll),
                                                 np.degrees(roll_dot),
                                                 np.degrees(yaw_dot))
    wx_1, wy_1, wz_1 = p, q, r
    omega_lab, omega_body, _, _ = FlightAnalysis.get_angular_velocities(x_frames, y_frames, z_frames, start_frame=0,
                                                                        end_frame=N, sampling_rate=1)
    omega_lab, omega_body = np.radians(omega_lab), np.radians(omega_body)
    wx_2, wy_2, wz_2 = omega_body[:, 0], omega_body[:, 1], omega_body[:, 2]

    # plt.plot(omega_body)
    # plt.plot(omega_lab, linestyle='--')
    # plt.show()
    # Plot all values in one plot

    plt.title("yaw, pitch, roll")
    plt.plot(yaw, label='yaw')
    plt.plot(pitch, label='pitch')
    plt.plot(roll, label='roll')
    plt.legend()
    plt.show()

    plot_pqr = True
    plot_est_omega = True
    plt.figure(figsize=(12, 8))
    if plot_pqr:
        plt.plot(wx_1, label='p -> wx', c='b')
        plt.plot(wy_1, label='q -> wy', c='r')
        plt.plot(wz_1, label='r -> wz', c='g')
    if plot_est_omega:
        plt.plot(wx_2, label='est wx', linestyle='--', c='b')
        plt.plot(wy_2, label='est wy', linestyle='--', c='r')
        plt.plot(wz_2, label='est wz', linestyle='--', c='g')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocities: wx, wy, wz')
    plt.legend()
    plt.show()

    omega = np.column_stack((p, q, r)) * 50
    # omega = omega_body * 500
    Visualizer.visualize_rotating_frames(x_frames, y_frames, z_frames, omega)
    pass


def get_omegas(csv_path):
    df = pd.read_csv(csv_path)
    df_dir1, df_dir2 = filter_between_light_dark(df)
    no_dark = get_3D_attribute_from_df(df_dir1)
    with_dark = get_3D_attribute_from_df(df_dir2)
    return no_dark, with_dark


def filter_between_light_dark(df):
    df_dir1 = df[df['wingbit'].str.startswith('dir1')]
    df_dir2 = df[df['wingbit'].str.startswith('dir2')]
    return df_dir1, df_dir2


def get_3D_attribute_from_df(df, attirbutes=["body_wx_take", "body_wy_take", "body_wz_take"]):
    wx = df[attirbutes[0]].values
    wx = wx[~np.isnan(wx)]
    wy = df[attirbutes[1]].values
    wy = wy[~np.isnan(wy)]
    wz = df[attirbutes[2]].values
    wz = wz[~np.isnan(wz)]
    omega = np.column_stack((wx, wy, wz))
    return omega, wx, wy, wz


def extract_yaw_pitch(vector):
    # Extract components
    v_x, v_y, v_z = vector

    # Calculate yaw angle
    yaw = np.arctan2(v_y, v_x)

    # Calculate pitch angle
    pitch = np.arcsin(v_z / np.linalg.norm(vector))

    # Convert from radians to degrees
    yaw_degrees = np.degrees(yaw)
    pitch_degrees = np.degrees(pitch)

    return yaw_degrees, pitch_degrees


def reconstruct_vector(yaw_degrees, pitch_degrees):
    # Convert angles from degrees to radians
    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)

    # Create rotation for yaw around z-axis
    r_yaw = R.from_euler('z', yaw, degrees=False)

    # Create rotation for pitch around y-axis
    r_pitch = R.from_euler('y', pitch, degrees=False)

    # Apply rotations to the original unit vector (1, 0, 0)
    initial_vector = np.array([1, 0, 0])
    rotated_vector = r_pitch.apply(r_yaw.apply(initial_vector))

    return rotated_vector


def scratch():
    vector = np.array([0.81, -0.53, -0.2])
    vector /= np.linalg.norm(vector)
    # Convert angles from degrees to radians

    yaw = np.arctan2(vector[1], vector[0])
    pitch = np.arctan2(vector[2], np.sqrt(vector[0] ** 2 + vector[1] ** 2))
    pitch_ = np.arcsin(vector[2])

    # vector = np.array([1,0,0])
    # yaw = -np.radians(30)
    # pitch = -np.radians(18.6)

    # Reconstruct the original vector from yaw and pitch
    reconstructed_vector = np.array([
        np.cos(-pitch) * np.cos(yaw),
        np.cos(-pitch) * np.sin(yaw),
        -np.sin(-pitch)
    ])

    yaw_degrees, pitch_degrees = extract_yaw_pitch(vector)



def compute_yaw_pitch(vec_bad):
    if vec_bad[0] < 0:
        vec_bad *= -1
    only_xy = vec_bad[[0, 1]] / np.linalg.norm(vec_bad[[0, 1]])
    yaw = np.rad2deg(np.arctan2(only_xy[1], only_xy[0]))
    pitch = np.rad2deg(np.arcsin(vec_bad[2]))

    # print(f"yaw: {yaw}, pitch: {pitch}")
    return yaw, pitch


def display_good_vs_bad_haltere(good_haltere, bad_haltere):
    no_dark, with_dark = get_omegas(bad_haltere)
    omega_light, wx_light, wy_light, wz_light = no_dark

    omega_good, wx_good, wy_good, wz_good = get_3D_attribute_from_df(pd.read_csv(good_haltere))
    omega_bad, wx_bad, wy_bad, wz_bad = get_3D_attribute_from_df(pd.read_csv(bad_haltere))

    # let us take only the omegas of the cut fly without dark
    omega_bad = omega_light

    mahal_dist_bad = calculate_mahalanobis_distance(omega_bad)
    omega_bad = omega_bad[mahal_dist_bad < 3]

    mahal_dist_good = calculate_mahalanobis_distance(omega_good)
    omega_good = omega_good[mahal_dist_good < 3]

    vec_good, yaw_good, pitch_good, yaw_std_good, pitch_std_good = get_pca_points(omega_good)
    vec_bad, yaw_bad, pitch_bad, yaw_std_bad, pitch_std_bad = get_pca_points(omega_bad)

    p1_good, p2_good = omega_good.mean(axis=0) + 10000 * vec_good, omega_good.mean(axis=0) - 10000 * vec_good
    p1_bad, p2_bad = omega_bad.mean(axis=0) + 10000 * vec_bad, omega_bad.mean(axis=0) - 10000 * vec_bad

    # Define body axis quivers
    size = 5000
    quivers = [
        {'x': [0, size], 'y': [0, 0], 'z': [0, 0], 'color': 'red', 'name': 'xbody'},
        {'x': [0, 0], 'y': [0, size], 'z': [0, 0], 'color': 'green', 'name': 'ybody'},
        {'x': [0, 0], 'y': [0, 0], 'z': [0, size], 'color': 'orange', 'name': 'zbody'}
    ]

    # Scatter plot for omega_good and omega_bad
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=omega_good[:, 0], y=omega_good[:, 1], z=omega_good[:, 2],
        mode='markers',
        marker=dict(size=1, color='red'),
        name='omega_good'
    ))

    fig.add_trace(go.Scatter3d(
        x=omega_bad[:, 0], y=omega_bad[:, 1], z=omega_bad[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='omega_bad'
    ))

    # Line for the bad axis
    fig.add_trace(go.Scatter3d(
        x=[p1_bad[0], p2_bad[0]], y=[p1_bad[1], p2_bad[1]], z=[p1_bad[2], p2_bad[2]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Bad Axis'
    ))

    # Add body axis quivers
    for q in quivers:
        fig.add_trace(go.Scatter3d(
            x=q['x'], y=q['y'], z=q['z'],
            mode='lines+text',
            line=dict(color=q['color'], width=5),
            name=q['name']
        ))

    # Update layout with formatted title
    fig.update_layout(
        scene=dict(
            xaxis_title='wx',
            yaxis_title='wy',
            zaxis_title='wz',
            aspectmode='data'
        ),
        title=f'Yaw of bad axis: {yaw_bad:.2f} (±{yaw_std_bad:.2f}), Pitch: {pitch_bad:.2f} (±{pitch_std_bad:.2f})',
        legend=dict(itemsizing='constant')
    )

    # Show plot
    # fig.show()

    # Save the figure to an HTML file
    fig.write_html("3d_plot.html")


def display_omegas_plt(omega_bad, omega_good, p1_bad, p2_bad, pitch_bad, yaw_bad):
    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(omega_good[:, 0], omega_good[:, 1], omega_good[:, 2], s=1, color='red')
    ax.scatter(omega_bad[:, 0], omega_bad[:, 1], omega_bad[:, 2], s=1, color='blue')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("wx")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    # plt.plot([p1_good[0], p2_good[0]], [p1_good[1], p2_good[1]], [p1_good[2], p2_good[2]], color='red')
    plt.plot([p1_bad[0], p2_bad[0]], [p1_bad[1], p2_bad[1]], [p1_bad[2], p2_bad[2]], color='blue')
    p1_body_axis = 5000 * np.array([1, 0, 0])
    p2_body_axis = 5000 * np.array([-1, 0, 0])
    # plt.plot([p1_body_axis[0], p2_body_axis[0]], [p1_body_axis[1], p2_body_axis[1]], [p1_body_axis[2], p2_body_axis[2]], color='black')
    size = 5000
    ax.quiver(0, 0, 0, size, 0, 0, color='r', label='xbody')
    ax.quiver(0, 0, 0, 0, size, 0, color='g', label='ybody')
    ax.quiver(0, 0, 0, 0, 0, size, color='orange', label='zbody')
    ax.legend()
    ax.title.set_text(f'Yaw of bad axis: {yaw_bad} and pitch is {pitch_bad}')
    ax.set_aspect('equal')
    plt.show()


def estimate_bootstrap_error(omegas):
    n_points = omegas.shape[0]
    yaw_samples = []
    pitch_samples = []
    num_bootstrap = 1000
    for _ in range(num_bootstrap):
        # Resample the point cloud with replacement
        resampled_points = omegas[np.random.choice(n_points, n_points, replace=True)]

        # Compute the principal component
        principal_component = get_first_component(resampled_points)

        # Normalize the principal component
        principal_component = principal_component / np.linalg.norm(principal_component)

        # Compute yaw and pitch
        yaw, pitch = compute_yaw_pitch(principal_component)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
    yaw_samples, pitch_samples = np.array(yaw_samples), np.array(pitch_samples)
    pitch_std = np.std(pitch_samples)
    yaw_std = np.std(yaw_samples)
    return yaw_std, pitch_std


def estimate_monte_carlo_error(omegas):
    num_samples = 1000
    sigma_matrix = np.array([[80, 80, 80]])  # sigma found
    yaw_samples, pitch_samples = [], []
    for _ in range(num_samples):
        mean_points = omegas
        mean_shape = mean_points.shape
        random_samples = np.random.randn(*mean_shape)  # Match the shape of mean_points
        new_omegas = mean_points + random_samples * sigma_matrix
        principal_component = get_first_component(new_omegas)
        yaw, pitch = compute_yaw_pitch(principal_component)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
    yaw_std = np.std(yaw_samples)
    pitch_std = np.std(pitch_samples)
    return yaw_std, pitch_std

def get_pca_points(omegas):
    first_component = get_first_component(omegas)
    yaw, pitch = compute_yaw_pitch(first_component)
    mean = np.mean(omegas, axis=0)
    yaw_std, pitch_std = estimate_bootstrap_error(omegas)
    # yaw_std, pitch_std = estimate_monte_carlo_error(omegas)
    return first_component, yaw, pitch, yaw_std, pitch_std


def get_first_component(omega):
    pca = PCA(n_components=3)
    pca.fit(omega)
    first_component = pca.components_[0]
    return first_component


def calculate_mahalanobis_distance(data):
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = np.array([mahalanobis(point, mean, inv_cov_matrix) for point in data])
    return distances


def display_omegas_dark_vs_light(csv_file):
    no_dark, with_dark = get_omegas(csv_file)
    omega_dark, wx_dark, wy_dark, wz_dark = with_dark
    omega_light, wx_light, wy_light, wz_light = no_dark
    all_omegas = np.concatenate((omega_dark, omega_light), axis=0)

    # remove outliers using mahalanobis
    mahal_dist_dark = calculate_mahalanobis_distance(omega_dark)
    omega_dark = omega_dark[mahal_dist_dark < 4]
    mahal_dist_light = calculate_mahalanobis_distance(omega_light)
    omega_light = omega_light[mahal_dist_light < 4]
    mahal_dist_all = calculate_mahalanobis_distance(all_omegas)
    all_omegas = all_omegas[mahal_dist_all < 3]

    vec_dark, yaw_dark, pitch_dark, yaw_std_dark, pitch_std_dark = get_pca_points(omega_dark)
    vec_light, yaw_light, pitch_light, yaw_std_light, pitch_std_light = get_pca_points(omega_light)
    vec_all, yaw_all, pitch_all, yaw_std_all, pitch_std_all = get_pca_points(all_omegas)

    pca = PCA(n_components=3)
    pca.fit(all_omegas)
    first_component = pca.components_[0]
    mean = pca.mean_
    p1 = mean + 10000 * first_component
    p2 = mean - 10000 * first_component

    # r = R.from_euler('y', -45, degrees=True)
    # Rot = np.array(r.as_matrix())
    # omega_dark = (Rot @ omega_dark.T).T
    # omega_light = (Rot @ omega_light.T).T

    omega_dist = np.array([all_omegas[i] @ first_component for i in range(len(all_omegas))])
    omega_light_dist = np.array([omega_light[i] @ first_component for i in range(len(omega_light))])
    # plt.hist(omega_dist, bins=100)
    # plt.hist(omega_light_dist, bins=100)
    # plt.show()

    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_omegas[:, 0], all_omegas[:, 1], all_omegas[:, 2], s=1, color='blue')

    ax.scatter(omega_light[:, 0], omega_light[:, 1], omega_light[:, 2], s=1, color='red')
    ax.scatter(omega_dark[:, 0], omega_dark[:, 1], omega_dark[:, 2], s=1, color='blue')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("wx")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    # plt.plot([p1_good[0], p2_good[0]], [p1_good[1], p2_good[1]], [p1_good[2], p2_good[2]], color='red')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue')
    p1_body_axis = 5000 * np.array([1, 0, 0])
    p2_body_axis = 5000 * np.array([-1, 0, 0])
    plt.plot([p1_body_axis[0], p2_body_axis[0]], [p1_body_axis[1], p2_body_axis[1]], [p1_body_axis[2], p2_body_axis[2]], color='black')
    size = 5000
    ax.quiver(0, 0, 0, size, 0, 0, color='r', label='xbody')
    ax.quiver(0, 0, 0, 0, size, 0, color='g', label='ybody')
    ax.quiver(0, 0, 0, 0, 0, size, color='orange', label='zbody')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()




if __name__ == '__main__':
    bad_haltere = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_severed_haltere.csv"
    good_haltere = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_good_haltere.csv"
    display_good_vs_bad_haltere(good_haltere, bad_haltere)
    # display_omegas_dark_vs_light(bad_haltere)
