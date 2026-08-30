# This file is modified from NeRF++: https://github.com/Kai-46/nerfplusplus

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    pass


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def get_camera_frustum_opengl_coord(H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0., 1., 0.])):
    '''X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    '''
    hfov = np.rad2deg(np.arctan(W / 2. / fx) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / fy) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array([[0., 0., 0., 1.0],                          # frustum origin
                               [-half_w, half_h,  -frustum_length, 1.0],   # top-left image corner
                               [half_w, half_h,   -frustum_length, 1.0],   # top-right image corner
                               [half_w, -half_h,  -frustum_length, 1.0],   # bottom-right image corner
                               [-half_w, -half_h, -frustum_length, 1.0]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])  # (8, 2)
    frustum_colors = np.tile(color.reshape((1, 3)), (frustum_lines.shape[0], 1))  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors


def draw_camera_frustum_geometry(c2ws, H, W, fx=600.0, fy=600.0, frustum_length=0.5,
                                 color=np.array([29.0, 53.0, 87.0])/255.0, draw_now=False, coord='opengl'):
    '''
    :param c2ws:            (N, 4, 4)  np.array
    :param H:               scalar
    :param W:               scalar
    :param fx:              scalar
    :param fy:              scalar
    :param frustum_length:  scalar
    :param color:           None or (N, 3) or (3, ) or (1, 3) or (3, 1) np array
    :param draw_now:        True/False call o3d vis now
    :return:
    '''
    N = c2ws.shape[0]

    num_ele = color.flatten().shape[0]
    if num_ele == 3:
        color = color.reshape(1, 3)
        color = np.tile(color, (N, 1))

    frustum_list = []
    if coord == 'opengl':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opengl_coord(H, W, fx, fy,
                                                                W2C=np.linalg.inv(c2ws[i]),
                                                                frustum_length=frustum_length,
                                                                color=color[i]))
    else:
        print('Undefined coordinate system. Exit')
        exit()

    frustums_geometry = frustums2lineset(frustum_list)

    if draw_now:
        o3d.visualization.draw_geometries([frustums_geometry])

    return frustums_geometry  # this is an o3d geometry object.


def extract_camera_position(extrinsic_matrix):
    """
    Extract camera position from the extrinsic matrix.
    The camera position in world coordinates is -R.T @ t
    """
    R = extrinsic_matrix[:3, :3]  # Rotation matrix
    t = extrinsic_matrix[:3, 3]   # Translation vector
    camera_position = -R.T @ t
    return camera_position


def create_axes_from_extrinsics(extrinsic_matrix, axis_length=0.1):
    """
    Create the camera axes in world coordinates using the extrinsic matrix.
    """
    R = extrinsic_matrix[:3, :3]  # Rotation matrix
    
    # Local coordinate system unit vectors
    x_axis_local = np.array([axis_length, 0, 0], dtype=np.float32)
    y_axis_local = np.array([0, axis_length, 0], dtype=np.float32)
    z_axis_local = np.array([0, 0, axis_length], dtype=np.float32)
    
    # Transform these axes using the rotation matrix
    x_axis_world = R @ x_axis_local
    y_axis_world = R @ y_axis_local
    z_axis_world = R @ z_axis_local
    
    # Extract camera position
    camera_position = extract_camera_position(extrinsic_matrix)
    
    # Return the points representing the axes
    return torch.stack([camera_position,
                        camera_position + x_axis_world,
                        camera_position,
                        camera_position + y_axis_world,
                        camera_position,
                        camera_position + z_axis_world])


def save_multi_camera_off_from_extrinsics(extrinsics, filename, axis_length=0.1):
    vertices = []
    edges = []
    
    # Loop over each extrinsic matrix
    for i in range(extrinsics.shape[0]):
        # Get the camera axes vertices
        axes_points = create_axes_from_extrinsics(extrinsics[i, ...], axis_length)
        
        # Add vertices to the list
        start_index = len(vertices)
        vertices.extend(axes_points)
        
        # Add edges to represent the axes
        edges.append((start_index, start_index + 1))  # X axis
        edges.append((start_index + 2, start_index + 3))  # Y axis
        edges.append((start_index + 4, start_index + 5))  # Z axis
    
    # Save to OFF file
    with open(filename, 'w') as file:
        file.write("OFF\n")
        file.write(f"{len(vertices)} 0 {len(edges)}\n")
        
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        for edge in edges:
            file.write(f"2 {edge[0]} {edge[1]}\n")


def save_off(points, filename):
    points = np.array(points)
    n_vertices = points.shape[0]

    with open(filename, 'w') as file:
        # OFF header
        file.write("OFF\n")

        # write the number of vertices, faces, and edges
        file.write(f"{n_vertices} 0 0\n")

        # save point
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

