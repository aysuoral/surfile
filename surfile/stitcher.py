"""
'surfile.stitcher'
- implementation of surface stitching methods

@author: Andrea Giura
"""
import copy
from functools import wraps

from matplotlib import patches, cm

from surfile import surface, funct, cutter
from scipy import optimize, signal, ndimage, interpolate

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import open3d as o3d
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

import pickle

def to_numpy(item):
            if isinstance(item, np.ndarray):
                return item
            
            if isinstance(item, o3d.geometry.PointCloud):
                print(f'[INFO STITCH] Auto-Converted {type(item)} to ndarray')
                return np.asarray(item.points)
            
            if hasattr(item, 'getPoints'):
                print(f'[INFO STITCH] Auto-Converted {type(item)} to ndarray')
                return np.asarray(item.getPoints(exclude_nan=True))
            
            print(f'[WARN STITCH] Could not ensure ndarray from type {type(item)}')
            return item

def ensure_numpy_pcd(func):
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, list):
            processed_data = [to_numpy(x) for x in data]
        else:
            processed_data = to_numpy(data)
            
        return func(processed_data, *args, **kwargs)
        
    return wrapper

def ensure_o3d_pc(func):
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        def to_o3d(item):
            if isinstance(item, o3d.geometry.PointCloud):
                return item
            
            if isinstance(item, np.ndarray):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(item)
                print(f'[INFO STITCH] Auto-Converted {type(item)} to o3d_pcd')
                return pc
            
            if hasattr(item, 'getPoints'):
                points = item.getPoints(exclude_nan=True)
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(np.asarray(points))
                print(f'[INFO STITCH] Auto-Converted {type(item)} to o3d_pcd')
                return pc
            
            print(f'[WARN STITCH] Could not ensure o3d PC from type {type(item)}')
            return item

        if isinstance(data, list):
            processed_data = [to_o3d(x) for x in data]
        else:
            processed_data = to_o3d(data)
            
        return func(processed_data, *args, **kwargs)
        
    return wrapper

@ensure_numpy_pcd
def pcd_to_surface(pcds: list[np.ndarray], dx, dy, force_same_size=True, bplt=False) -> surface.Surface:
    """
    Transforms pc into a Surface object.
    
    1. Fits a least-squares plane.
    2. Projects points onto the plane coordinate system.
    3. Interpolates onto a regular grid defined by dx, dy.
    """
    
    z_maps = []

    for pcd in pcds:
        points = pcd

        # # Plane equation: ax + by + d = z  => [x, y, 1][a, b, d]^T = z
        # A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        # C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        # a, b, d = C 
        
        # normal = np.array([-a, -b, 1.0])
        # normal /= np.linalg.norm(normal)
        # print(f"normal1: {normal}")

        normal = pcd_least_squared_plane(pcds)
        print(f"normal2: {normal}")

        
        z_axis = np.array([0, 0, 1])
        v = np.cross(normal, z_axis)
        c = np.dot(normal, z_axis)
        s = np.linalg.norm(v)
        
        if True:  # Already aligned
            R = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        
        centroid = np.mean(points, axis=0)
        centered_pts = points - centroid
        rotated_pts = centered_pts @ R.T
        
        x_pts = rotated_pts[:, 0]
        y_pts = rotated_pts[:, 1]
        z_pts = rotated_pts[:, 2] # These are now distances from the plane
        
        x_min, x_max = x_pts.min(), x_pts.max()
        y_min, y_max = y_pts.min(), y_pts.max()
        n_x = int(np.ceil((x_max - x_min) / dx))
        n_y = int(np.ceil((y_max - y_min) / dy))
        grid_x = np.linspace(x_min, x_min + n_x * dx, n_x)
        grid_y = np.linspace(y_min, y_min + n_y * dy, n_y)
        gx, gy = np.meshgrid(grid_x, grid_y)

        print("\n--- GRID INFO ---")
        print(f"x_min: {x_min:.3f}, x_max: {x_max:.3f}, range: {x_max - x_min:.3f}")
        print(f"y_min: {y_min:.3f}, y_max: {y_max:.3f}, range: {y_max - y_min:.3f}")

        print(f"n_x: {n_x}, n_y: {n_y}")
        print(f"dx: {dx}, dy: {dy}")

        print(f"grid_x shape: {grid_x.shape}, grid_y shape: {grid_y.shape}")
        print(f"meshgrid shape: gx: {gx.shape}, gy: {gy.shape}")
        print("------------------\n")
        
        z_sum, _, _ = np.histogram2d(y_pts, x_pts, bins=[grid_y, grid_x], weights=z_pts)
        
        # 3. Calculate the count of points in each bin
        counts, _, _ = np.histogram2d(y_pts, x_pts, bins=[grid_y, grid_x])
                
        # Average the bins and fill NaNs
        z_sum = np.divide(z_sum, counts, out=np.zeros_like(z_sum), where=counts!=0)
        mean_val = np.nanmean(z_pts)
        z_sum[counts == 0] = mean_val
        
        print(f'[INFO PCD_TO_SUR] Could not bin {z_sum[counts == 0].size} elements')

        # Create the coordinate map (the "query" points in index space)
        # Since gx and gy are already spaced by dx/dy, their index-space is just a ramp
        coords = np.array([
            (gy - y_min) / dy, 
            (gx - x_min) / dx
        ])
        
        print(f'[INFO PCD_TO_SUR] Converting pc using spacings dx: {dx:.3f} um, dy: {dy:.3f} um')
        # order=3 is equivalent to cubic interpolation
        z_map = ndimage.map_coordinates(z_sum, coords, order=1, mode='nearest')

        z_maps.append(z_map)

    if force_same_size == True:
        min_rows = min(z_map.shape[0] for z_map in z_maps)
        min_cols = min(z_map.shape[1] for z_map in z_maps)

        for i in range(len(z_maps)):
            z_maps[i] = z_maps[i][:min_rows, :min_cols]

    surfs = []

    for z_map in z_maps:
        surf = surface.Surface()
        surf.setValues(dx, dy, z_map, bplt=bplt)
        surfs.append(surf)

    return surfs

@ensure_numpy_pcd
def pcd_to_o3d_pcd(pcd: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    
    return pc

class TransformParams:
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    tz: float
    
    def __init__(self):
        pass
    
    @classmethod
    def from_numbers(cls, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        instance = cls()
        instance.rx, instance.ry, instance.rz, instance.tx, instance.ty, instance.tz = rx, ry, rz, tx, ty, tz
        return instance
        
    @classmethod
    def from_list(cls, params: list[float]):
        # params order: x y z rx ry rz
        instance = cls()
        instance.tx, instance.ty, instance.tz, instance.rx, instance.ry, instance.rz = params
        return instance

    @classmethod
    def from_tuples(cls, rot: R, trasl: np.ndarray):
        instance = cls()
        eul = rot.as_euler('xyz')
        instance.rx, instance.ry, instance.rz = eul[0], eul[1], eul[2]
        instance.tx, instance.ty, instance.tz = trasl[0], trasl[1], trasl[2]
        return instance
    
    @classmethod
    def from_file(cls, filename, tr_n, header=1):
        instance = cls()
        with open(filename, "r") as f:
            riga = list(f)[header + tr_n]
        riga = riga[3:].strip().replace(',', ' ')
        values = list(map(float, riga.split()))
        instance.tx, instance.ty, instance.tz, instance.rx, instance.ry, instance.rz = values
        return instance
    
    def from_kabsch(m: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the least-squares roto-translation that maps points m → f.

        Parameters
        ----------
        m : (N, 3)  moving  points
        f : (N, 3)  fixed   points

        Returns
        -------
        tx, ty, tz : float
            Translation components along x, y, z axes.
        rx, ry, rz : float
            Rotation angles (in degrees) around x, y, z axes (XYZ convention).

        After the transform:
            f_approx = (R @ m.T).T + t

        Info
        ----
        Roto-translation (rigid body) alignment of 3 moving points to 3 fixed points.
        Algorithm: Kabsch (1976) — minimises RMSD via SVD on the cross-covariance matrix.

        The resulting rotation matrix is converted to Euler angles (XYZ order, degrees),
        and the translation vector is split into its components.

        No scaling is applied (pure rotation + translation).
        """
        m = np.asarray(m, dtype=float)
        f = np.asarray(f, dtype=float)
        assert m.shape == f.shape and m.ndim == 2 and m.shape[1] == 3

        # 1. Centroid subtraction
        cm = m.mean(axis=0)
        cf = f.mean(axis=0)
        m_c = m - cm
        f_c = f - cf

        # 2. Cross-covariance matrix  H = m_centred^T · f_centred
        H = m_c.T @ f_c          # (3, 3)

        # 3. SVD
        U, S, Vt = np.linalg.svd(H)

        # 4. Correct for reflection (det check ensures a proper rotation, det = +1)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])   # d = sign(det); flips last singular vector if needed

        # 5. Rotation and translation
        R_m = Vt.T @ D @ U.T

        t = cf - R_m @ cm

        # 6. Homogeneous 4×4 matrix
        T = np.eye(4)
        T[:3, :3] = R_m
        T[:3,  3] = t
        
        # from rotation matrix to angles
        r = R.from_matrix(R_m)
        rx, ry, rz  = r.as_euler('xyz', degrees=True)
    

        # from traslation vector to traslations

        tx, ty, tz = t

        return tx, ty, tz, rx, ry, rz
    
    def get_params(self):
        return [self.rx, self.ry, self.rz, self.tx, self.ty, self.tz]
    
    def get_matrix(self):
        Rmat = R.from_euler('xyz', np.radians([self.rx, self.ry, self.rz])).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = [self.tx, self.ty, self.tz]
        return T
    
    def rescale(self, factor):
        self.tx *= factor
        self.ty *= factor
        self.tz *= factor
    
    def __str__(self):
        return f"TransformParams: {self.get_matrix()}"

@ensure_numpy_pcd
def apply_transform(points: np.ndarray, params: TransformParams | np.ndarray, params0: TransformParams | np.ndarray=None):
    """
    Applies a transformation on the points, if params0 is provided
    performs the transformation relative to the 0 transformation
    """
    if hasattr(params, 'get_matrix'):
        T = params.get_matrix()
    else: T = params
    
    if params0 is not None:
        if hasattr(params0, 'get_matrix'):
            T0 = params0.get_matrix()
        else: T0 = params0

        T0_inv = np.linalg.inv(T0)
        
        T = T0_inv @ T
        
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]

@ensure_o3d_pc
def remove_outliers_from_point_cloud(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    # TODO: maybe write your own outlier function?
    pc, _ = point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    return np.asarray(pc.points)

@ensure_numpy_pcd
def pcd_least_squared_plane(pcds: list[np.ndarray]):
    normals = []
    for pcd in pcds:
        points = pcd 
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        a, b, d = C 
        
        normal = np.array([-a, -b, 1.0])
        normal /= np.linalg.norm(normal)

        normals.append(normal)
        
    return normals

def merge_and_downsample_point_cloud(pc1: np.ndarray, pc2: np.ndarray, voxel_size=0.001):
    combined = np.vstack([pc1, pc2])
    pc = pcd_to_o3d_pcd(combined)
    pc_down = pc.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pc_down.points)

@ensure_o3d_pc
def show_point_cloud(point_clouds: list[o3d.geometry.PointCloud], colors="normal"):
    if colors is not None:
        point_clouds = assign_defined_colors_to_point_clouds(point_clouds, colors)
    o3d.visualization.draw_geometries(point_clouds, point_show_normal=False)

@ensure_o3d_pc
def assign_defined_colors_to_point_clouds(point_clouds: list[o3d.geometry.PointCloud], colors: None | str = None):
    """
    Given a list of pc and a list of colors paints uniform color the pc, if the colors are not given assigns automatically a color to each pc

    Parameters
    ----------
    point_clouds : list[o3d.geometry.PointCloud]
        the pcs
    colors : list | None | str
        The colors, can be strings, rgb tuples, rgb vectors, color hex string, None or "normal" to color based on the point cloud's normals.
    """
    # per colorare in base alle normali
    if colors == "normal":
        for i, pc in enumerate(point_clouds):
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=3))
            # pc.orient_normals_consistent_tangent_plane(10)
            # pc.orient_normals_to_align_with_direction([1, 0, 0])
            normals = np.asarray(pc.normals)

            colors = (normals + 1) / 2  # da [-1,1] a [0,1]
            pc.colors = o3d.utility.Vector3dVector(colors)
        return point_clouds

    elif colors == "uniform":
        num_pcs = len(point_clouds)
        cmap = plt.get_cmap("tab10")  # pastel1, pastel2, Accent
        colors = [cmap(i % 10) for i in range(num_pcs)]

        for i, pc in enumerate(point_clouds):
            raw_color = colors[i % len(colors)]
            rgb_color = np.asarray(mcolors.to_rgb(raw_color))
            
            pc.paint_uniform_color(rgb_color)

        return point_clouds
    
    else:
        for i, pc in enumerate(point_clouds):
            z = np.asarray(pc.points)[:, 2]
            n1 = (z - z.min()) / (z.max() - z.min())
            p = np.asarray(pc.points)
            n2 = np.linalg.norm(p - np.mean(p, axis=0), axis=1) 
            n2 = (n2 - n2.min()) / (n2.max() - n2.min())
            cc = plt.get_cmap(colors)(n2 * n1)[:, :3]
            pc.colors = o3d.utility.Vector3dVector(cc)
        
        return point_clouds

class Isolator():
    geometrical: str = 'geometrical'
    maxmin: str = 'maxmin'
    KDTree: str = 'KDTree'

    type: str

    def __init__(self, type: str, stitchprc=80, max_distance=None, axes='xyz'):
        self.type = type
        self.stitchprc = stitchprc
        self.max_distance = max_distance
        self.axes = axes

    def apply_isolator(self, fixed_pts: np.ndarray, moving_pts: np.ndarray, bplt=False):
        if self.type == 'geometrical': return self.isolate_common_points_geometrical(fixed_pts, moving_pts, self.stitchprc, bplt)
        elif self.type == 'maxmin': return self.isolate_common_points_max_min(fixed_pts, moving_pts, self.axes, bplt)
        elif self.type == 'KDTree': return self.isolate_common_points_kdtree(fixed_pts, moving_pts, self.max_distance, bplt)

        else:
            raise ValueError('Unknown isolator type')
    
    @staticmethod
    def plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts): 
        show_point_cloud([fixed_subset, moving_subset], colors=None)

    @staticmethod
    def isolate_common_points_geometrical(fixed_pts: np.ndarray, moving_pts: np.ndarray, stitchprc=80, bplt=False):
        fixed_center = np.mean(fixed_pts, axis=0)
        moving_center = np.mean(moving_pts, axis=0)

        # direzione movimento
        moving_dir = moving_center - fixed_center
        norm = np.linalg.norm(moving_dir)
        if norm == 0:
            moving_dir = np.array([1.0, 0.0, 0.0])
        else:
            moving_dir /= norm

        # filtra punti entro stitchprc
        dist_fixed = (fixed_pts - fixed_center) @ moving_dir
        fixed_subset = fixed_pts[dist_fixed >= norm * (1 - stitchprc / 100)]
        
        dist_moving = (moving_pts - moving_center) @ moving_dir
        moving_subset = moving_pts[norm * (1 - stitchprc / 100) <= -dist_moving]

        if bplt: Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)
        
        return fixed_subset, moving_subset

    @staticmethod
    def isolate_common_points_max_min(fixed_pts: np.ndarray, moving_pts: np.ndarray, axes: str, bplt=False):
        x_mM, y_mM, z_mM = get_common_boundries(fixed_pts, moving_pts)

        if 'x' not in axes: x_mM = [-np.inf, +np.inf]
        if 'y' not in axes: y_mM = [-np.inf, +np.inf]
        if 'z' not in axes: z_mM = [-np.inf, +np.inf]

        make_mask = lambda pts: (
            (pts[:, 0] >= x_mM[0]) & (pts[:, 0] <= x_mM[1]) &
            (pts[:, 1] >= y_mM[0]) & (pts[:, 1] <= y_mM[1]) &
            (pts[:, 2] >= z_mM[0]) & (pts[:, 2] <= z_mM[1])
        )

        fixed_subset = fixed_pts[make_mask(fixed_pts)]
        moving_subset = moving_pts[make_mask(moving_pts)]
            
        if bplt: Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)

        return fixed_subset, moving_subset
    
    @staticmethod
    def isolate_common_points_kdtree(fixed_pts: np.ndarray, moving_pts: np.ndarray, max_distance: float=None, bplt=False):
        A_to_B = lambda A, B: cKDTree(B).query(A, k=1)

        dist_f2m, _ = A_to_B(fixed_pts, moving_pts)
        dist_m2f, _ = A_to_B(moving_pts, fixed_pts)

        if max_distance is None:
            dist_hist, dist_bins = np.histogram(np.hstack((dist_f2m, dist_m2f)), 50)

            max_distance = dist_bins[np.nanargmax(dist_hist) + 1]

            if bplt:
                fig, ax = plt.subplots()

                ax.hist(dist_f2m, bins=50, alpha=0.5, label="fixed → moving")
                ax.hist(dist_m2f, bins=50, alpha=0.5, label="moving → fixed")

                ax.hist(np.hstack((dist_f2m, dist_m2f)), bins=50, alpha=0.5, label="all")
                ax.vlines([max_distance], 0, np.nanmax(dist_hist), label='max distance')

                ax.set_xlabel("Distance")
                ax.set_ylabel("Count")

        fixed_subset = fixed_pts[dist_f2m <= max_distance]
        moving_subset = moving_pts[dist_m2f <= max_distance]

        ax = None

        if bplt:
            Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)
        return fixed_subset, moving_subset, ax
 
    @staticmethod
    def isolate_manual(left_pcd: np.ndarray, right_pcd: np.ndarray):

        def pick_point(points, window_name):
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(points)

            print(f"\n{window_name}")
            print("Select the point with Shift + left click")
            print("Remove the last selected point with Shift + right")

            pcd_o3d = assign_defined_colors_to_point_clouds([pcd_o3d], colors="normal")[0]

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name=window_name)
            vis.add_geometry(pcd_o3d)

            render_option = vis.get_render_option()
            render_option.point_size = 0.5

            vis.run()
            vis.destroy_window()

            picked = vis.get_picked_points()

            return points[picked], picked

        fixed_subset, left_idx = pick_point(left_pcd, "Left point cloud")
        moving_subset, right_idx = pick_point(right_pcd, "Right point cloud")

        print(f"\033[95mSelected point left =\033[0m",
               np.array2string(fixed_subset, formatter={'float_kind': lambda x: f"{x:.8f}"}))
        print()
        print(f"\033[96mSelected point right =\033[0m",
               np.array2string(moving_subset, formatter={'float_kind': lambda x: f"{x:.8f}"}))

        return fixed_subset, moving_subset

class Thresholder():
    type: str

    value: str = 'value'
    sphere: str = 'sphere'
    cuboid: str = 'cuboid'
    KDTree: str = 'KDTree'

    def __init__(self, type: str, val=20, threshold_expansion=1.2):
        self.type = type
        self.threshold_expansion = threshold_expansion
        self.val = val

    def apply_thresholder(self, fixed_subset, moving_subset):
        if self.type == 'value': return self.threshold_value(self.val)
        elif self.type == 'sphere': return self.threshold_sphere(fixed_subset, moving_subset)
        elif self.type == 'cuboid': return self.threshold_cuboid(fixed_subset, moving_subset, self.threshold_expansion)
        elif self.type == 'KDTree': return self.threshold_KDTree(fixed_subset, moving_subset, self.threshold_expansion)
        else:
            raise ValueError('Unknown thresholder type')
    
    @staticmethod    
    def threshold_value(x): return x
    
    @staticmethod
    def threshold_sphere(fixed_subset, moving_subset):
        c_fixed = np.mean(fixed_subset, axis=0)
        c_moving = np.mean(moving_subset, axis=0)

        r_fixed = ...
        r_moving = ...
    
    @staticmethod
    def threshold_cuboid(fixed_subset, moving_subset, threshold_expansion=3):
        x_mM, y_mM, z_mM = get_common_boundries(fixed_subset, moving_subset)

        x_overlap = max(0, x_mM[1] - x_mM[0])
        y_overlap = max(0, y_mM[1] - y_mM[0])
        z_overlap = max(0, z_mM[1] - z_mM[0])

        volume = x_overlap * y_overlap * z_overlap
        threshold = threshold_expansion * (volume ** (1/3))

        return threshold
    
    @staticmethod
    def threshold_KDTree(fixed_subset, moving_subset, threshold_expansion=1.2):
        diffs = KDTree_mutual_diffs(fixed_subset, moving_subset)
        threshold = np.mean(np.abs(diffs))
        threshold_dev = np.std(np.abs(diffs))

        return threshold * threshold_expansion

def get_common_boundries(pts_a, pts_b):
    min_a = np.min(pts_a, axis=0)
    max_a = np.max(pts_a, axis=0)

    min_b = np.min(pts_b, axis=0)
    max_b = np.max(pts_b, axis=0)

    x_common = (max(min_a[0], min_b[0]), min(max_a[0], max_b[0]))   
    y_common = (max(min_a[1], min_b[1]), min(max_a[1], max_b[1]))
    z_common = (max(min_a[2], min_b[2]), min(max_a[2], max_b[2]))

    return x_common, y_common, z_common

def KDTree_mutual_diffs(fixed_points, moving_points):
    fixed_tree = cKDTree(fixed_points)
    moving_tree = cKDTree(moving_points)

    dist_f2m, idx_f2m = moving_tree.query(fixed_points, k=1, workers= -1)
    dist_m2f, idx_m2f = fixed_tree.query(moving_points, k=1, workers= -1)

    mask = (np.arange(len(fixed_points)) == idx_m2f[idx_f2m])
    if not np.any(mask):
        return float('inf')
    
    diffs = fixed_points[mask] - moving_points[idx_f2m[mask]]
    return diffs

def _composeFigure(left, right, T, R=None, support=None, sp=20):
    """
    Compose the stitched image (stitch in x direction)

    Parameters
    ----------
    left : np.array
        The left figure
    right : np.array
        The right figure
    T : List
        The translation vector
    R : np.array
        The rotation matrix
    sp : int
        The overlap of the 2 images %

    Returns
    -------
    composed : np.array
        The composed array
    """
    lcopy = copy.deepcopy(left)
    rcopy = copy.deepcopy(right)
    
    if R is not None and support is not None:  # add rotation displacement
        beta = R[0, 2]
        alpha = R[2, 1]
        lcopy += -beta * support[0] + alpha * support[1]

    print(f'[INFO] {T=}, {R=}')

    # patches creation
    sp = int(lcopy.shape[1] * (sp / 100))
    lcopy = np.roll(lcopy, shift=(T[0], T[1]), axis=(1, 0))  # add x, y displacements

    lcopy = lcopy[:, :-sp // 2]
    rcopy = rcopy[:, sp // 2:]

    if T[2] == 'best': lcopy -= np.mean(lcopy[:, -1]) - np.mean(rcopy[:, 0])

    st = np.hstack((lcopy, rcopy))

    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1)
    ax.imshow(lcopy[:, -sp:])
    bx.imshow(rcopy[:, :sp])

    fig2, cx = plt.subplots(nrows=1, ncols=1)
    cx.imshow(st, cmap=cm.viridis)
    plt.show()

@ensure_numpy_pcd
def rescale_point_cloud(point_clouds: list[np.ndarray], scales=None, revert=False):
    """
    Rescale a list of point clouds along x, y, and z axes.

    If `scales` is not provided, the scale factors are computed from the
    first point cloud in the list as:

        scales = maxs - mins

    where `mins` and `maxs` are the per-axis minimum and maximum values of
    the reference point cloud. The same scale factors are then applied to
    all point clouds in the list.

    If `revert` is True, the inverse scaling is applied by using `1 / scales`.

    Parameters
    ----------
    point_clouds : list[np.ndarray]
        List of point clouds, each of shape (N, 3).
    scales : np.ndarray or None, optional
        Array of shape (3,) containing the scale factors for x, y, and z.
        If None, the factors are computed from the first point cloud.
    revert : bool, optional
        If True, apply the inverse scaling factors instead of the direct
        scaling factors. Default is False.

    Returns
    -------
    scaled_list : list[np.ndarray]
        List of rescaled point clouds.
    scales : np.ndarray
        The scale factors that were used.

    Notes
    -----
    This function only divides coordinates by the scale factors.
    It does not shift the point clouds by subtracting their minimum values.
    Therefore, the output is not guaranteed to lie in the [0, 1] range.
    """
    if scales is None:
        ref = point_clouds[0]

        mins = ref.min(axis=0)
        maxs = ref.max(axis=0)
        scales = maxs - mins

    scaled_list = []
    scales = 1 / scales if revert else scales

    for pts in point_clouds:
        pts_scaled = pts / scales
        scaled_list.append(pts_scaled)

    return scaled_list, scales

class SurfaceStitcher:
    @staticmethod
    def stitchCorrelation2(surl, surr, stitchPrc=20, samplingPrc=50, correlateDer=True, bplt=False):
        """
        Finds the best allignment between surl and surr
        by calculating the maximum of the cross correlation
        
        Parameters
        ----------
        samplingPrc : int
            The percentage of the points of the overimposed
            surfaces that is sampled from the arrays
        surl : surface.Surface
            The left image to be stitched
        surr : surface.Surface
            The right image to be stitched
        stitchPrc : int
            the percentage of the image overlapping
        correlateDer : bool
            If true uses the first derivatice for the cross correlation to
            in order to compare the slope of the sample instead of the height
        bplt : bool
            If true plots the stitched image
        """
        if surl.Z.shape != surr.Z.shape:
            raise ValueError("[ERROR COR] surl and surr must have the same shape for FGR stitching")

        len = int(surl.Z.shape[1] * stitchPrc / 100)
        
        # find the interested zones to be stitched
        lZone = copy.deepcopy(surl.Z[:, -len:])
        rZone = copy.deepcopy(surr.Z[:, :len])
        
        print(f'[INFO COR] {lZone.shape=}, {rZone.shape=}')

        if correlateDer:
            lZone = np.diff(lZone)
            rZone = np.diff(rZone)

        # take a central patch from the second image
        center_x, center_y = lZone.shape[0] // 2, lZone.shape[1] // 2
        size_x, size_y = lZone.shape[0] * samplingPrc // 100, lZone.shape[1] * samplingPrc // 100
        sampleL = lZone[center_x - size_x // 2: center_x + size_x // 2,
                  center_y - size_y // 2: center_y + size_y // 2]
        sampleR = rZone[center_x - size_x // 2: center_x + size_x // 2,
                  center_y - size_y // 2: center_y + size_y // 2]
        

        # correlate the patch with the first image to find its position
        nrmze = lambda a: a / np.linalg.norm(a)
        ccL = signal.correlate2d(lZone, sampleR, mode='valid')
        ccR = signal.correlate2d(rZone, sampleL, mode='valid')

        ML = np.argmax(ccL)
        yML, xML = np.unravel_index(ML, ccL.shape)
        print(f'[INFO COR] {ML=} {xML=} {yML=}')

        MR = np.argmax(ccR)
        yMR, xMR = np.unravel_index(MR, ccR.shape)
        print(f'[INFO COR] {MR=} {xMR=} {yMR=}')
        bestLTranslation = [ccL.shape[1] // 2 - xML, ccL.shape[0] // 2 - yML]
        bestRTranslation = [ccR.shape[1] // 2 - xMR, ccR.shape[0] // 2 - yMR]

        meanTranslation = [(bestLTranslation[i] - bestRTranslation[i]) // 2 for i in [0, 1]]
        print(f'[INFO COR] {bestLTranslation=}\n{bestRTranslation=}\n{meanTranslation=}')

        flippedccR = np.flip(ccR)
        cross_cc = ccL * flippedccR
        M = np.argmax(cross_cc)
        yM, xM = np.unravel_index(M, cross_cc.shape)
        bestMeanTranslation = [cross_cc.shape[1] // 2 - xM, cross_cc.shape[0] // 2 - yM]
        print(f'\n\n{M=} {xM=} {yM=}')
        print(f'{bestMeanTranslation=}')

        if bplt:
            fig, ((ax, bx, cx), (dx, ex, fx)) = plt.subplots(nrows=2, ncols=3)
            ax.imshow(ccL)
            ax.set_title('ccL')
            ax.plot(xML, yML, 'ro', ms=5)
            bx.imshow(lZone)
            cx.imshow(sampleR)

            dx.imshow(ccR)
            dx.set_title('ccR')
            dx.plot(xMR, yMR, 'ro', ms=5)
            ex.imshow(rZone)
            fx.imshow(sampleL)
            funct.persFig([ax, bx, cx, dx, ex, fx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')

            plt.get_current_fig_manager().full_screen_toggle()

            fig2, (lx, mx, nx) = plt.subplots(nrows=1, ncols=3)
            lx.imshow(ccL)
            mx.imshow(np.flip(ccR))
            nx.imshow(ccL * np.flip(ccR))
            funct.persFig([lx, mx, nx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')
            nx.plot(xM, yM, 'r.', ms=5)

            plt.show()

            _composeFigure(surl.Z, surr.Z,
                           T=[bestMeanTranslation[0], bestMeanTranslation[1], 'best'],
                           sp=stitchPrc)

    @staticmethod
    @ensure_numpy_pcd
    def stitchManual(point_clouds: list[np.ndarray], radius=5,  bplt=False):

        def get_patch(cloud, center, r):
            dists = np.linalg.norm(cloud - center, axis=1)
            print(f"get_patch found {len(mask := (cloud[dists < r]))} points near selected point ({center})")
            mean_patch_point = np.mean(mask, axis=0)
            return mean_patch_point
        
        def optimize(fixed, moving):
            fp, mp = Isolator.isolate_manual(fixed, moving)

            # if len(fp) != len(mp):
            #     raise RuntimeError("[ERROR MANUAL] Select same amount of points from left and right")
            
            # if len(fp) < 3:
            #     print("[ERROR MANUAL] You need to select at least 3 points!")


            # # calc traslatıon

            # centroid_fixed = fp.mean(axis=0)
            # centroid_moving = mp.mean(axis=0)

            # translation = centroid_fixed - centroid_moving

            # print(f"translation = {translation}")

            # normal_f, normal_m = pcd_least_squared_plane([fp, mp])
            # print(f"[INFO MANUAL] {normal_f=} {normal_m=}")

            # # calc rotatıon

            # # rotation = normal_f - normal_m

            # # rotation = np.arctan2(normal_m, normal_f)
            # # rotation_deg = np.rad2deg(rotation)
            # # print(f"Rotation = {rotation}")
            # # print(f"Rotation deg = {rotation_deg}")

            # rotation = rotations_from_normals(normal_f, normal_m) 
            
            if len(fp) != len(mp):
                raise RuntimeError("[ERROR MANUAL] Select same amount of points from left and right")
            
            if len(fp) < 3:
                print("[ERROR MANUAL] You need to select at least 3 points!")


            fixed_patches = []
            moving_patches = []

            for pf, pm in zip(fp, mp):
                fixed_patches.append(get_patch(fixed, pf, radius))
                moving_patches.append(get_patch(moving, pm, radius))

            fp = np.vstack(fixed_patches)
            mp = np.vstack(moving_patches)
           
            tx, ty, tz, rx, ry, rz = TransformParams.from_kabsch(mp, fp)          


            RTM = TransformParams.from_numbers(tx, ty, tz, rx, ry, rz)

            moved = apply_transform(moving, RTM)

            # apply transformatıon
            return moved


        fixed = np.asarray(point_clouds[0])
        point_clouds_T = []

        for i, pc in enumerate(point_clouds[1:]):            
            moving = np.asarray(pc)
            moved = optimize(fixed, moving)
            point_clouds_T.append(moved)

            fixed = np.vstack([fixed, moved])

        if bplt: 
            show_point_cloud([fixed])
            show_point_cloud([point_clouds[0]] + point_clouds_T, colors=None)

        # saving in a pickle file:
        with open("point_clouds.pkl", "wb") as f:
            pickle.dump((fixed, point_clouds_T), f)

        
        return fixed, point_clouds_T

    @staticmethod
    @ensure_numpy_pcd
    def stitchRobot(point_clouds: list[np.ndarray], robotTfile, bplt=False):
        """
        Finds the best allignment between surl and surr
        by using the robot positions and rotations recorded in robotTfile

        Parameters
        ----------
        surfaces : list[surface.Surface]
            The list of surfaces to be stitched, in the order they were acquired
        robotTfile : str
            The path of the file containing the robot positions and rotations
        """
        robot_trans = []
        point_clouds_clean = []
        
        for i, pc in enumerate(point_clouds):
            pc = remove_outliers_from_point_cloud(pc)
            point_clouds_clean.append(pc)
            
            tr = TransformParams.from_file(robotTfile, i, header=1)
            tr.rescale(1000)
            robot_trans.append(tr)
            
        print(f"[INFO ROBOT STITCH] Loaded {len(robot_trans)} robot transformations for {len(point_clouds_clean)} surfaces")
        
        point_clouds_T = []
        fixed_ref = point_clouds_clean[0]
        for pc, trasf in zip(point_clouds_clean, robot_trans):
            pts = pc
            pts_T = apply_transform(pts, trasf, params0=robot_trans[0])
            point_clouds_T.append(pts_T)
            fixed_ref = merge_and_downsample_point_cloud(fixed_ref, pts_T)
        
        if bplt: 
            show_point_cloud([fixed_ref])
            show_point_cloud(point_clouds_T, colors=None)
        
        return fixed_ref, point_clouds_T    

    @staticmethod
    @ensure_numpy_pcd
    def stitchRMSE(point_clouds_T: list[np.ndarray], n_calls, isolator: Isolator, bplt=False):
        """
        Finds the best alignment between transformed point clouds
        by minimizing the RMSE between mutually matched points
        in the overlapping regions.

        Parameters
        ----------
        point_clouds_T : list[np.ndarray]
            List of transformed point clouds, in the order they
            will be stitched together
        n_calls : int
            Number of optimization calls used by gp_minimize
            during the RMSE minimization
        isolator : callable
            Function that takes fixed_pts and moving_pts and returns
            the overlapping subsets to be compared
        bplt : bool
            If true plots the RMSE and number of matched points
            during each optimization, and shows the final stitched
            point cloud

        Returns
        -------
        fixed_pc : np.ndarray
            The final stitched point cloud after sequentially
            aligning and merging all point clouds
        """
        def optimize(fixed_pts, moving_pts):
            U_tx, U_ty, U_tz = 55.2, 60.6, 69.3  
            U_theta = 0.5  

            # tx ty tz rx ry rz
            t0 = [0, 0, 0, 0, 0, 0]

            def objective(x):
                p = TransformParams.from_list(x)
                moved = apply_transform(moving_pts, p)

                fixed_sub, moved_sub = isolator.apply_isolator(fixed_pts, moved, bplt=False)
                diffs = KDTree_mutual_diffs(fixed_sub, moved_sub)  # just use the dıfference, not the mutual kdtree
                rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
                
                npoints.append(len(diffs))
                rmses.append(rmse)
                return rmse

            space = [
                Real(t0[0] - U_tx, t0[0] + U_tx),
                Real(t0[1] - U_ty, t0[1] + U_ty),
                Real(t0[2] - U_tz, t0[2] + U_tz),
                Real(t0[3] - U_theta, t0[3] + U_theta),
                Real(t0[4] - U_theta, t0[4] + U_theta),
                Real(t0[5] - U_theta, t0[5] + U_theta),
            ]

            res = gp_minimize(objective, space, x0=t0, n_calls=n_calls, random_state=42)

            best = res.x
            best_p = TransformParams.from_list(best)
            
            aligned = apply_transform(moving_pts, best_p)

            return aligned, res
        
        fixed = np.asarray(point_clouds_T[0])

        for i, pc in enumerate(point_clouds_T[1:]):
            rmses = []
            npoints = []
            
            moving = np.asarray(pc)
            print(f'[INFO RMSE] Optimizing image {i}')
            optimized_moving, _ = optimize(fixed, moving)

            fixed = np.vstack([fixed, optimized_moving])

            if bplt:
                fig, (ax, bx) = plt.subplots(2, 1)
                ax.set_title(f'Image {i} opt')
                
                ax.plot(range(1, n_calls+1), rmses)
                ax.set_xlabel("Number of call")
                ax.set_ylabel("RMSE")
                ax.grid(True)

                bx.plot(range(1, n_calls+1), npoints)
                bx.set_xlabel("Number of call")
                bx.set_ylabel("npoints")
                bx.grid(True)

        fixed_pc = fixed

        if bplt:
            show_point_cloud([fixed_pc], colors=None)

        return fixed_pc
    
    @staticmethod
    @ensure_numpy_pcd
    def stitchICP(point_clouds_T: list[np.ndarray], thresholder: Thresholder, isolator: None | Isolator, bplt=False):
        """
        Refines the alignment of transformed point clouds using ICP.

        The first point cloud is taken as the fixed reference. Each subsequent
        Iterative Closest Point (ICP), then merged into the final stitched
        point cloud is aligned to the accumulated fixed point cloud by using
        result. If an isolator function is provided, ICP is applied only on the
        isolated overlapping regions, while the resulting transformation is
        applied to the full moving point cloud.

        Parameters
        ----------
        point_clouds_T : list[np.ndarray]
            List of transformed point clouds to be stitched, in the order they
            were acquired. Each point cloud is expected to be an array of shape
            (N, 3).
        threshold : float
            Maximum correspondence distance used by ICP.
        isolator : callable, optional
            Function that takes ``fixed_pts`` and ``moving_pts`` as input and
            returns ``fixed_subset, moving_subset`` to restrict ICP to a common
            region. If None, ICP is applied to the full point clouds.
        bplt : bool, optional
            If True, displays the final stitched point cloud.

        Returns
        -------
        fixed_pc : np.ndarray
            Final stitched point cloud obtained after sequential ICP alignment
            and merging.

        Notes
        -----
        This method assumes that the input point clouds are already roughly
        aligned, for example by a prior robot-based transformation. ICP is then
        used only as a refinement step.
        """
        def optimize(fixed_pts, moving_pts):

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_pts, moving_pts)
            else:
                fixed_subset, moving_subset = fixed_pts, moving_pts

            pc_fixed = pcd_to_o3d_pcd(fixed_subset)
            pc_moving = pcd_to_o3d_pcd(moving_subset)

            trans_init = np.eye(4)
            threshold = thresholder.apply_thresholder(fixed_subset, moving_subset)
            print(f'{threshold=}')

            reg_p2p = o3d.pipelines.registration.registration_icp(
                pc_moving, pc_fixed, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
            
            aligned = apply_transform(moving_pts, reg_p2p.transformation)

            return aligned, reg_p2p
        
        fixed = np.asarray(point_clouds_T[0])

        for i, pc in enumerate(point_clouds_T[1:]):
            moving = np.asarray(pc)

            print(f"[INFO ICP] Optimizing image {i}")

            optimized_moving, _ = optimize(fixed, moving)

            fixed = np.vstack([fixed, optimized_moving])

        fixed_pc = fixed

        if bplt:
            show_point_cloud([fixed_pc])

        return fixed_pc

    @staticmethod
    @ensure_numpy_pcd
    def stitchFGR(point_clouds_T: list[np.ndarray], voxel_size: float, thresholder: Thresholder, isolator: None | Isolator, bplt=False):

        def optimize(fixed_pts, moving_pts):
            [fixed_scaled, moving_scaled], scales = rescale_point_cloud([fixed_pts, moving_pts])

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_scaled, moving_scaled, bplt=bplt)
            else:
                fixed_subset, moving_subset = fixed_scaled, moving_scaled

            pc_fixed = pcd_to_o3d_pcd(fixed_subset)
            pc_moving = pcd_to_o3d_pcd(moving_subset)

            pc_fixed_down = pc_fixed.voxel_down_sample(voxel_size)
            pc_moving_down = pc_moving.voxel_down_sample(voxel_size)

            radius_normal = voxel_size * 2
            pc_fixed_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )
            pc_moving_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )

            radius_feature = voxel_size * 5 
            fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pc_fixed_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pc_moving_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            threshold = thresholder.apply_thresholder(fixed_subset, moving_subset)
            print(f'{threshold=}')
            reg_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                pc_moving_down,
                pc_fixed_down,
                moving_fpfh,
                fixed_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=threshold
                )
            )


            aligned_scaled = apply_transform(moving_scaled, reg_fgr.transformation)  

            [aligned], _ = rescale_point_cloud(
                [aligned_scaled],
                scales=scales,
                revert=True
            )

            return aligned

        fixed = np.asarray(point_clouds_T[0])

        for i, pc in enumerate(point_clouds_T[1:]):
            moving = np.asarray(pc)
            print(f"[INFO FGR] Optimizing image {i}")
            optimized_moving = optimize(fixed, moving)
            fixed = np.vstack([fixed, optimized_moving])

        if bplt:
            show_point_cloud([fixed])

        return fixed

    @staticmethod
    @ensure_numpy_pcd
    def stitchCorrelation(point_clouds_T: list[np.ndarray], dx: float, dy: float, isolator: None | Isolator, samplingPrc, correlateDer=True, bplt=False):

        def optimize(fixed_pts, moving_pts):

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_pts, moving_pts, bplt=bplt)
            else:
                fixed_subset, moving_subset = fixed_pts, moving_pts

            fixed_surf, moving_surf = pcd_to_surface([fixed_subset, moving_subset], dx, dy, True, bplt=True)

            plt.show()

            print('shapes:', fixed_surf.Z.shape, moving_surf.Z.shape)

            if fixed_surf.Z.shape != moving_surf.Z.shape:   
                raise ValueError("[ERROR COR] fixed_surf and moving_surf must have the same shape.")

            lzone = copy.deepcopy(fixed_surf.Z)
            rzone = copy.deepcopy(moving_surf.Z)

            print(f"[INFO COR] {lzone.shape=}, {rzone.shape=}")

            if correlateDer:
                lzone = np.diff(lzone)
                rzone = np.diff(rzone)  

            shift, _, _ = phase_cross_correlation(
                lzone,
                rzone,
                upsample_factor=10
            )

            shift_y, shift_x = shift

            tx = shift_x * dx
            ty = shift_y * dy

            if bplt:
                fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
                ax.imshow(lzone)
                ax.set_title("lzone")
                bx.imshow(rzone)
                bx.set_title("rzone")
                funct.persFig([ax, bx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')
                plt.show()



            # center_x, center_y = lzone.shape[0] // 2, lzone.shape[1] // 2
            # size_x = lzone.shape[0] * samplingPrc // 100
            # size_y = lzone.shape[1] * samplingPrc // 100

            # get_sample = lambda arr: arr[
            # center_x - size_x // 2 : center_x + size_x // 2,
            # center_y - size_y // 2 : center_y + size_y // 2
            # ]

            # sampleL = get_sample(lzone)
            # sampleR = get_sample(rzone)



        #     ccL = signal.correlate2d(lzone, sampleR, mode='valid')
        #     ccR = signal.correlate2d(rzone, sampleL, mode='valid')

        #     ML = np.argmax(ccL)
        #     yML, xML = np.unravel_index(ML, ccL.shape)
        #     print(f"[INFO COR] {ML=} {xML=} {yML=}")

        #     MR = np.argmax(ccR)
        #     yMR, xMR = np.unravel_index(MR, ccR.shape)
        #     print(f"[INFO COR] {MR=} {xMR=} {yMR=}")

        #     bestLTranslation = [ccL.shape[1] // 2 - xML, ccL.shape[0] // 2 - yML]
        #     bestRTranslation = [ccR.shape[1] // 2 - xMR, ccR.shape[0] // 2 - yMR]

        #     meanTranslation = [(bestLTranslation[i] - bestRTranslation[i]) // 2 for i in [0, 1]]
        #     print(f"[INFO COR] {bestLTranslation=}")
        #     print(f"[INFO COR] {bestRTranslation=}")
        #     print(f"[INFO COR] {meanTranslation=}")

        #     flippedccR = np.flip(ccR)
        #     cross_cc = ccL * flippedccR

        #     M = np.argmax(cross_cc)
        #     yM, xM = np.unravel_index(M, cross_cc.shape)
        #     bestMeanTranslation = [cross_cc.shape[1] // 2 - xM, cross_cc.shape[0] // 2 - yM]

        #     print(f"[INFO COR] {M=} {xM=} {yM=}")
        #     print(f"[INFO COR] {bestMeanTranslation=}")

        #     tx = bestMeanTranslation[0] * dx
        #     ty = bestMeanTranslation[1] * dy

            temp = moving_pts.copy()
            temp[:, :2] += [tx, ty]

            mask_f = (fixed_pts[:, 0] >= temp[:, 0].min()) & (fixed_pts[:, 0] <= temp[:, 0].max()) & (fixed_pts[:, 1] >= temp[:, 1].min()) & (fixed_pts[:, 1] <= temp[:, 1].max())
            mask_m = (temp[:, 0] >= fixed_pts[:, 0].min()) & (temp[:, 0] <= fixed_pts[:, 0].max()) & (temp[:, 1] >= fixed_pts[:, 1].min()) & (temp[:, 1] <= fixed_pts[:, 1].max())
            tz = np.median(fixed_pts[mask_f, 2]) - np.median(temp[mask_m, 2])
            # if isolator != None:
            #     fix_sub, temp_sub = isolator.apply_isolator(fixed_pts, temp, bplt=bplt)

            # tz = np.median(fix_sub[:, 2]) - np.median(temp_sub[:, 2])

            aligned = moving_pts.copy()
            aligned[:, 0] += tx
            aligned[:, 1] += ty
            aligned[:, 2] += tz

            print(f"before mean z = {np.mean(moving_pts[:, 2])}")
            print(f"after mean z  = {np.mean(aligned[:, 2])}")

            # if bplt:
            #     fig, ((ax, bx, cx), (gx, ex, fx)) = plt.subplots(nrows=2, ncols=3)
            #     ax.imshow(ccL)
            #     ax.set_title('ccL')
            #     ax.plot(xML, yML, 'ro', ms=5)
            #     bx.imshow(lzone)
            #     bx.set_title('lzone')
            #     cx.imshow(sampleR)
            #     cx.set_title('rsample')

            #     gx.imshow(ccR)
            #     gx.set_title('ccR')
            #     gx.plot(xMR, yMR, 'ro', ms=5)
            #     ex.imshow(rzone)
            #     ex.set_title('rzone')
            #     fx.imshow(sampleL)
            #     fx.set_title('lsample')
            #     funct.persFig([ax, bx, cx, gx, ex, fx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')

            #     fig2, (lx, mx, nx) = plt.subplots(nrows=1, ncols=3)
            #     lx.imshow(ccL)
            #     lx.set_title('ccL')
            #     mx.imshow(np.flip(ccR))
            #     mx.set_title('ccR rot')
            #     nx.imshow(ccL * np.flip(ccR))
            #     nx.set_title('cc mult')
            #     funct.persFig([lx, mx, nx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')
            #     nx.plot(xM, yM, 'r.', ms=5)

            #     plt.show()

            return aligned, [tx, ty]

        


        fixed = np.asarray(point_clouds_T[0])

        for i, pc in enumerate(point_clouds_T[1:]):
            moving = np.asarray(pc)
            print(f"[INFO COR] Optimizing image {i}")
            optimized_moving, _ = optimize(fixed, moving)
            fixed = np.vstack([fixed, optimized_moving])

        if bplt:
            show_point_cloud([fixed])

        return fixed