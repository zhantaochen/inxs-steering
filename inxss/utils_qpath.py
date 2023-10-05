import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_Sqw_func(coordinates, values, sigma=0):
    func_list = []
    if values.ndim == 2:
        values = values[None,...]
    # if sigma > 0:
    #     values = scipy.ndimage.gaussian_filter(values, sigma, axes=(0,))
        
    for i in range(values.shape[0]):
        func_list.append(RegularGridInterpolator(coordinates, values[i], method='linear'))
    
    return func_list

# Function created by OpenAI's GPT-4 model
def linspace_2D(points, N_sep):
    points = np.asarray(points)
    N_pt = len(points)
    
    if N_pt < 2:
        raise ValueError("At least two points are required.")
        
    # Preallocate the array size
    N = (N_pt - 1) * N_sep + 1
    result = np.ones((N, 2), dtype=float)
    
    for i in range(N_pt - 1):
        x_vals = np.linspace(points[i, 0], points[i + 1, 0], N_sep+1)[:-1]
        y_vals = np.linspace(points[i, 1], points[i + 1, 1], N_sep+1)[:-1]
        result[(i*N_sep):((i+1)*N_sep), 0] = x_vals
        result[(i*N_sep):((i+1)*N_sep), 1] = y_vals
        
        # print(i*N_sep, (i+1)*N_sep, np.stack((x_vals, y_vals), axis=-1), result[(i*N_sep):((i+1)*N_sep), :])
        result[(i*N_sep):((i+1)*N_sep), :] = np.stack((x_vals, y_vals), axis=-1)
    # Include the last point of the last segment
    result[-1] = points[-1]
    
    return result


# Function created by OpenAI's GPT-4 model
def linspace_2D_equidistant(points, N, return_indices=False):
    points = np.asarray(points, dtype=float)  # Ensure points are float type
    N_pt = len(points)

    if N_pt < 2:
        raise ValueError("At least two points are required.")

    # Calculate the total length of the path
    segment_lengths = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    total_length = np.sum(segment_lengths)

    # Preallocate the array size
    result = np.zeros((N, 2), dtype=float)  # Ensure result is float type

    # Determine how many points are needed for each segment
    N_segment = np.round(segment_lengths / total_length * (N-1)).astype(int)

    # Handle rounding error by adjusting the last segment
    N_segment[-1] = N - np.sum(N_segment[:-1]) - 1

    # Generate points for each segment
    idx = 0
    critical_point_indices = [0]  # Start with the first critical point
    for i in range(N_pt - 1):
        x_vals = np.linspace(points[i, 0], points[i + 1, 0], N_segment[i]+1)[:-1]
        y_vals = np.linspace(points[i, 1], points[i + 1, 1], N_segment[i]+1)[:-1]

        result[idx:idx+N_segment[i], :] = np.stack((x_vals, y_vals), axis=-1)
        idx += N_segment[i]
        critical_point_indices.append(idx)

    # Include the last point of the last segment
    result[-1] = points[-1]
    if return_indices:
        return result, critical_point_indices
    else:
        return result


def plot_points(points, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c='w')
    ax.plot(points[:, 0], points[:, 1], c='w', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    plt.show()