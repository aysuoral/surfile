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

def ensure_numpy_pcd(func):
    @wraps(func)
    def wrapper(data, *args, **kwargs):
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
def pcd_to_surface(pcd: np.ndarray, dx, dy, bplt=False) -> surface.Surface:
    """
    Transforms pc into a Surface object.
    
    1. Fits a least-squares plane.
    2. Projects points onto the plane coordinate system.
    3. Interpolates onto a regular grid defined by dx, dy.
    """
    points = pcd

    # Plane equation: ax + by + d = z  => [x, y, 1][a, b, d]^T = z
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b, d = C 
    
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    s = np.linalg.norm(v)
    
    if s < 1e-9:  # Already aligned
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
    
    z_sum, _, _ = np.histogram2d(x_pts, y_pts, bins=[grid_x, grid_y], weights=z_pts)
    
    # 3. Calculate the count of points in each bin
    counts, _, _ = np.histogram2d(x_pts, y_pts, bins=[grid_x, grid_y])
            
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
    
    surf = surface.Surface()
    surf.setValues(dx, dy, z_map, bplt=bplt)
    
    return surf

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
def apply_transform(points: np.ndarray, params: TransformParams, params0: TransformParams=None):
    """
    Applies a transformation on the points, if params0 is provided
    performs the transformation relative to the 0 transformation
    """
    T = params.get_matrix()
    
    if params0 is not None:
        # perform a relative transform
        T0 = params0.get_matrix()
        T0_inv = np.linalg.inv(T0)
        
        T = T0_inv @ T
        
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]



@ensure_o3d_pc
def remove_outliers_from_point_cloud(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    # TODO: maybe write your own outlier function?
    pc, _ = point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    return np.asarray(pc.points)

def merge_and_downsample_point_cloud(pc1: np.ndarray, pc2: np.ndarray, voxel_size=0.001):
    combined = np.vstack([pc1, pc2])
    pc = pcd_to_o3d_pcd(combined)
    pc_down = pc.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pc_down.points)

@ensure_o3d_pc
def show_point_cloud(point_clouds: list[o3d.geometry.PointCloud], uniform_colors=False):    
    if uniform_colors: 
        point_clouds = assign_defined_colors_to_point_clouds(point_clouds)
    o3d.visualization.draw_geometries(point_clouds)

@ensure_o3d_pc
def assign_defined_colors_to_point_clouds(point_clouds: list[o3d.geometry.PointCloud], colors: list | None = None):
    """
    Given a list of pc and a list of colors paints uniform color the pc, if the colors are not given assigns automatically a color to each pc

    Parameters
    ----------
    point_clouds : list[o3d.geometry.PointCloud]
        the pcs
    colors : list | None
        The colors, can be strings, rgb tuples, rgb vectors, color hex string, None
    """
    num_pcs = len(point_clouds)

    if colors is None:
        cmap = plt.get_cmap("tab10")  # pastel1, pastel2, Accent
        colors = [cmap(i % 10) for i in range(num_pcs)]

    if len(colors) < num_pcs:
        print(f"Warning: Only {len(colors)} colors provided for {num_pcs} point clouds. Cycling colors.")

    for i, pc in enumerate(point_clouds):
        raw_color = colors[i % len(colors)]
        rgb_color = np.asarray(mcolors.to_rgb(raw_color))
        
        pc.paint_uniform_color(rgb_color)

    return point_clouds

def isolate_common_points(fixed_points: np.ndarray, moving_points: np.ndarray, stitchprc, bplt=False):    
    fixed_center = np.mean(fixed_points, axis=0)
    moving_center = np.mean(moving_points, axis=0)
    
    # direzione movimento
    moving_dir = moving_center - fixed_center
    norm = np.linalg.norm(moving_dir)
    if norm == 0:
        moving_dir = np.array([1.0, 0.0, 0.0])
    else:
        moving_dir /= norm

    # filtra punti entro stitchprc
    dist_fixed = (fixed_points - fixed_center) @ moving_dir
    fixed_subset = fixed_points[dist_fixed <= norm * (1 - stitchprc / 100)]
    
    dist_moving = (moving_points - moving_center) @ moving_dir
    moving_subset = moving_points[norm * (1 - stitchprc / 100) <= -dist_moving]
    
    if bplt:
        show_point_cloud([
            surface_to_pcd(fixed_subset), 
            surface_to_pcd(moving_subset), 
            surface_to_pcd(fixed_points), 
            surface_to_pcd(moving_points)])
    return fixed_subset, moving_subset

def mutual_points_RMSE(fixed_points, moving_points):
    fixed_tree = cKDTree(fixed_points)
    moving_tree = cKDTree(moving_points)

    dist_f2m, idx_f2m = moving_tree.query(fixed_points, k=1, workers= -1)
    dist_m2f, idx_m2f = fixed_tree.query(moving_points, k=1, workers= -1)

    mask = (np.arange(len(fixed_points)) == idx_m2f[idx_f2m])
    if not np.any(mask):
        return float('inf')
    
    diffs = fixed_points[mask] - moving_points[idx_f2m[mask]]
    rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    return rmse



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

class SurfaceStitcher:
    @staticmethod
    def stitchCorrelation(surl, surr, stitchPrc=20, samplingPrc=50, correlateDer=True, bplt=False):
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
    def stitchFGR(surl, surr, stitchPrc=20):
        """
        Finds the best allignment between surl and surr
        by first registering approximatively the 2 images
        using a FGR feature matcher, and then improves
        the result by refining with an ICP (iterative closest point)
        registration

        ref: http://www.open3d.org/docs/0.9.0/python_api/open3d.registration.html
        FGR: http://vladlen.info/papers/fast-global-registration.pdf
        ICP: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=121791
        PFH: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5152473

        Parameters
        ----------
        surl : surface.Surface
            The left image to be stitched
        surr : surface.Surface
            The right image to be stitched
        stitchPrc : int
            the percentage of the image overlapping
        """
        if surl.Z.shape != surr.Z.shape:
            raise ValueError("[ERROR FGR] surl and surr must have the same shape for FGR stitching")

        len = int(surl.Z.shape[1] * stitchPrc / 100)
        
        # find the interested zones to be stitched
        lZone = copy.deepcopy(surl.Z[:, -len:])
        rZone = copy.deepcopy(surr.Z[:, :len])
        
        print(f'[INFO FGR] {lZone.shape=}, {rZone.shape=}')

        # scale parameter for normalization
        scale = 1

        scalez = np.max([lZone.max(), rZone.max()]) * 2 * scale  # re-range [-scale * 0.5, scale * 0.5]
        lZone /= scalez
        rZone /= scalez

        # scale xy max dimention
        if lZone.shape[1] > lZone.shape[0]:
            scaley = lZone.shape[0] * scale / lZone.shape[1]
            scalefactor = lZone.shape[1]

            X = np.linspace(0, scale, lZone.shape[1])
            Y = np.linspace(0, scaley, lZone.shape[0])
        else:
            scalex = lZone.shape[1] * scale / lZone.shape[0]
            scalefactor = lZone.shape[0]

            X = np.linspace(0, scalex, lZone.shape[1])
            Y = np.linspace(0, scale, lZone.shape[0])
        mesh_x, mesh_y = np.meshgrid(X, Y)

        def toPC(zone):
            xyz = np.zeros((np.size(mesh_x), 3))
            xyz[:, 0] = np.reshape(mesh_x, -1)
            xyz[:, 1] = np.reshape(mesh_y, -1)
            xyz[:, 2] = np.reshape(zone, -1)

            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            return pcd

        pcd_l = toPC(lZone)
        pcd_r = toPC(rZone)
        
        print(pcd_l)
        
        # exit()

        def draw_registration_result(source, target, transformation):
            source_temp = copy.deepcopy(source)
            target_temp = copy.deepcopy(target)
            source_temp.paint_uniform_color(np.array([139, 0, 0]) / 255)
            target_temp.paint_uniform_color(np.array([0, 0, 139]) / 255)
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target_temp])

        def preprocess_point_cloud(pcd, voxel_size):
            print("[INFO FGR] Downsample with a voxel size %.3f." % voxel_size)
            pcd_down = pcd.voxel_down_sample(voxel_size)

            radius_normal = voxel_size * 2
            print("[INFO FGR] Estimate normal with search radius %.3f." % radius_normal)
            pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

            radius_feature = voxel_size * 5
            print("[INFO FGR] Compute FPFH feature with search radius %.3f." % radius_feature)
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )
            return pcd_down, pcd_fpfh

        def prepare_dataset(voxel_size):
            source = pcd_l
            target = pcd_r

            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            
            # o3d.visualization.draw_geometries([source_down])
            # o3d.visualization.draw_geometries([target_down])
            
            return source, target, source_down, target_down, source_fpfh, target_fpfh

        def execute_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
            distance_threshold = voxel_size * 0.5
            print("[INFO FGR] FGR registration on downsampled point clouds.")
            print("[INFO FGR] downsampling voxel size is %.3f," % voxel_size)
            print("[INFO FGR] distance threshold %.3f." % distance_threshold)
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold)
            )
            return result

        voxel_size = 0.02 * scale
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

        result_fgr = execute_global_registration(source_down, target_down,
                                                 source_fpfh, target_fpfh,
                                                 voxel_size)
        print(result_fgr)
        print("[INFO FGR] Transformation is:")
        print(result_fgr.transformation)
        draw_registration_result(source_down, target_down, result_fgr.transformation)

        print("[INFO FGR] Refine with point-to-point ICP")
        # actually FGR should not need this step
        distance_threshold = 0.001 * scale
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold,
            init=result_fgr.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(True))
        print(reg_p2p)
        print("[INFO FGR] Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(source, target, reg_p2p.transformation)

        # Transformation matrix in the form:
        # [[ 1 , -si, +be, xt],
        #  [+si,  1 , -al, yt],
        #  [-be, +al,  1 , zt],
        #  [ 0 ,  0 ,  0 ,  1]]

        xtransl = int(reg_p2p.transformation[0, 3] * scalefactor)
        ytransl = int(reg_p2p.transformation[1, 3] * scalefactor)
        ztransl = reg_p2p.transformation[2, 3] * scalez
        _composeFigure(surl.Z, surr.Z,
                       T=[xtransl, ytransl, 'best'],
                       # R=reg_p2p.transformation[0:3, 0:3],  # this doesn't seem right
                       # support=(surl.X, surl.Y),
                       sp=stitchPrc)

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
            show_point_cloud(point_clouds_T, uniform_colors=True)
        
        return fixed_ref, point_clouds_T

    @staticmethod
    def stitchRMSE(point_clouds: list[np.ndarray], bplt=False):
        def optimize_alignment(fixed_points, moving_points):
            def objective(params: list):
                transformed = apply_transform(moving_points, TransformParams.from_list(params))
                fixed_subset, moving_subset = isolate_common_points(fixed_points, transformed)
                return mutual_points_RMSE(fixed_points, transformed)
            
            t_nom = [0,0,0,0,0,0]
            U_tx, U_ty, U_tz = 0.0552, 0.0606, 0.0693  # mm
            U_theta = 0.001  # gradi

            search_space = [
                Real(t_nom[0]-U_tx, t_nom[0]+U_tx),
                Real(t_nom[1]-U_ty, t_nom[1]+U_ty),
                Real(t_nom[2]-U_tz, t_nom[2]+U_tz),
                Real(t_nom[3]-U_theta, t_nom[3]+U_theta),
                Real(t_nom[4]-U_theta, t_nom[4]+U_theta),
                Real(t_nom[5]-U_theta, t_nom[5]+U_theta)]
            
            res = gp_minimize(objective, search_space, x0=t_nom, n_calls=50, random_state=42)
            
            aligned_points = apply_transform(moving_points, res.x)
            return aligned_points, res.x, float(res.fun)
            
        point_clouds_T = []
        fixed_ref = point_clouds[0].points
        for pc in point_clouds:
            pts = np.asarray(pc.points)
            pts_T, _, _ = optimize_alignment(fixed_ref_pc, pts)
            point_clouds_T.append(surface_to_pcd(pts_T))
            fixed_ref = merge_and_downsample_point_cloud(fixed_ref, pts_T)
        
        fixed_ref_pc = surface_to_pcd(fixed_ref)
        if bplt: 
            show_point_cloud([fixed_ref_pc])
            
        return fixed_ref_pc, point_clouds_T