"""
'surfile.stitcher'
- implementation of surface stitching methods

@author: Andrea Giura
"""
import copy

from matplotlib import patches, cm

from surfile import surface, funct, cutter
from scipy import optimize, signal, ndimage

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

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

def make_o3d_cloud(surf: surface.Surface | np.ndarray, color=None, remove_outliers=False) -> o3d.geometry.PointCloud:
    points = surf.getPoints(exclude_nan=True) if isinstance(surf, surface.Surface) else surf
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pc.paint_uniform_color(color)
        
    if remove_outliers:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    return pc

def merge_and_downsample_point_cloud(points1, points2, voxel_size=0.001):
    combined = np.vstack([points1, points2])
    pc = make_o3d_cloud(combined)
    pc_down = pc.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pc_down.points)

def show_point_cloud(point_clouds: list[o3d.geometry.PointCloud], defined_colors=False, name="Point Cloud"):
    if defined_colors:
        colors = [np.array([139, 0, 0]) / 255, np.array([139, 139, 0]) / 255, np.array([0, 0, 139]) / 255, np.array([0, 139, 0]) / 255, np.array([139, 0, 139]) / 255, np.array([0, 139, 139]) / 255]
        for pc, color in zip(point_clouds, colors):
            pc.paint_uniform_color(color)
    
    o3d.visualization.draw_geometries(point_clouds)

def isolate_common_points(fixed_points_pc, moving_points_pc, stitchprc):
    fixed_points = np.asarray(fixed_points_pc.points)
    moving_points = np.asarray(moving_points_pc.points)
    
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
    fixed_subset = fixed_points[dist_fixed <= norm * stitchprc / 100]
    
    dist_moving = (moving_points - moving_center) @ moving_dir
    moving_subset = moving_points[norm * stitchprc / 100 <= -dist_moving]
    
    # show_point_cloud([make_o3d_cloud(fixed_subset), make_o3d_cloud(moving_subset), fixed_points_pc, moving_points_pc], defined_colors=True)
    return make_o3d_cloud(fixed_subset), make_o3d_cloud(moving_subset)

def apply_transform(params: TransformParams, points, params0: TransformParams=None):
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
    def stitchRobot(surfaces: list[surface.Surface], robotTfile, bplt=False):
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
        def read_robot_positions_rotations(file_robot):
            tras: TransformParams = []
            with open(file_robot, "r") as f:
                f.readline() # skip first line
                for riga in f:
                    riga = riga[3:].strip().replace(',', ' ')
                    
                    valori = list(map(float, riga.split()))
                    tr = TransformParams.from_list(valori)
                    tr.rescale(1000)  # convert to um
                    if len(valori) != 6:
                        continue
                    tras.append(tr)
            return tras
        
        point_clouds = []
        for surf in surfaces:
            pc = make_o3d_cloud(surf, remove_outliers=True)
            point_clouds.append(pc)
            
        robot_trans = read_robot_positions_rotations(robotTfile)
        print(f"[INFO ROBOT STITCH] Loaded {len(robot_trans)} robot transformations for {len(point_clouds)} surfaces")
        
        point_clouds_T = []
        fixed_ref = point_clouds[0].points
        for pc, trasf in zip(point_clouds, robot_trans):
            pts = np.asarray(pc.points)
            pts_T = apply_transform(trasf, pts, params0=robot_trans[0])
            point_clouds_T.append(make_o3d_cloud(pts_T))
            fixed_ref = merge_and_downsample_point_cloud(fixed_ref, pts_T)
        
        fixed_ref_pc = make_o3d_cloud(fixed_ref)
        if bplt: 
            show_point_cloud([fixed_ref_pc], name="Stitched Point Cloud from Robot Poses", defined_colors=False)
            show_point_cloud(point_clouds_T, name="Stitched Point Cloud from Robot Poses", defined_colors=True)
        
        return fixed_ref_pc, point_clouds_T
        
        
        
        
        
        
        
    # @staticmethod
    # def stitchSSDminimize(surl, surr, stitchPrc=20, bplt=False):
    #     """
    #     Match image locations using SSD minimization.
    #
    #     Areas from `surl` are matched with areas from `surr`. These areas
    #     are defined as patches located around pixels with Gaussian
    #     weights.
    #
    #     https://scikit-image.org/docs/stable/auto_examples/registration/plot_stitching.html
    #
    #     Parameters
    #     ----------
    #     surl : surface.Surface
    #         The left image to be stitched
    #     surr : surface.Surface
    #         The right image to be stitched
    #     stitchPrc : int
    #         the percentage of the image overlapping
    #     bplt : bool
    #         If true plots the stitched image
    #
    #     Returns
    #     -------
    #     match_coords: (2, m) array
    #         The points in `coordsR` that are the closest corresponding matches to
    #         those in `coordsL` as determined by the (Gaussian weighted) sum of
    #         squared differences between patches surrounding each point.
    #     """
    #     lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
    #     # rZone = surr.Z[:, :int(surr.Z.shape[1] * (stitchPrc / 100))]
    #     rZone = lZone
    #
    #     samplingPxls = 40
    #     sampleSpacing = 50
    #     samplingSdev = 5
    #
    #     startx = starty = samplingPxls
    #     stopx = lZone.shape[0] - samplingPxls
    #     stopy = lZone.shape[1] - samplingPxls
    #
    #     coordsL = np.mgrid[startx:stopx:sampleSpacing, starty:stopy:sampleSpacing].reshape(2, -1).T
    #     coordsR = np.mgrid[startx:stopx:sampleSpacing, starty:stopy:sampleSpacing].reshape(2, -1).T
    #
    #     y, x = np.mgrid[-samplingPxls:samplingPxls + 1, -samplingPxls:samplingPxls + 1]
    #     weights = np.exp(-0.5 * (x ** 2 + y ** 2) / samplingSdev ** 2)
    #     weights /= 2 * np.pi * samplingSdev * samplingSdev
    #
    #     match_list = []
    #     for rL, cL in coordsL:
    #         roiL = lZone[rL - samplingPxls:rL + samplingPxls + 1, cL - samplingPxls:cL + samplingPxls + 1]
    #         roiR_list = [rZone[rR - samplingPxls:rR + samplingPxls + 1,
    #                      cR - samplingPxls:cR + samplingPxls + 1] for rR, cR in coordsR]
    #         # sum of squared differences
    #         ssd_list = [np.sum(weights * (roiL - roiR) ** 2) for roiR in roiR_list]
    #         match_list.append(coordsL[np.argmin(ssd_list)])
    #
    #     print(match_list)
    #     return np.array(match_list)
    
    # @staticmethod
    # def stitchMinimizeNorm(surl, surr, stitchPrc=20, pixelScan=40, bplt=False):
    #     """
    #     Given 2 surfaces finds the best allignement
    #     by minimizing the norm2 of the difference

    #     Parameters
    #     ----------
    #     surl : surface.Surface
    #         The left image to be stitched
    #     surr : surface.Surface
    #         The right image to be stitched
    #     stitchPrc : int
    #         the percentage of the image overlapping
    #     pixelScan: int
    #         the number of pixel the method tryes to displace the images
    #         an higher number results in longer excution time
    #     bplt : bool
    #         If true plots the stitching process, limits the radius scan to 5 pixels
    #         Use this only to see graphically and very slowly what this function does.

    #     Returns
    #     -------
    #     surface.Surface
    #         The stitched image
    #     """
    #     # Given starting displacement (0, 0) in x and y
    #     # move surr % of stitching.py over the other % surl
    #     # and minimize surr(x - a; y - b) - surl(x, y)

    #     # We need a function that given a, b moves surr
    #     # and subtracts surr moved from surl

    #     # find the interested zones to be stitched
    #     lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
    #     rZone = surl.Z[:, :int(surl.Z.shape[1] * (stitchPrc / 100))]
    #     # rZone = np.roll(lZone, 30, axis=1)  # used for testing

    #     ny, nx = lZone.shape

    #     if bplt:
    #         fig, (ax, bx, cx) = plt.subplots(nrows=1, ncols=3)
    #         plot_data = ax.imshow(lZone)
    #         bx.imshow(lZone)
    #         cx.imshow(rZone)
    #         rectb = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
    #         rectc = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
    #         bx.add_patch(rectb)
    #         cx.add_patch(rectc)

    #         funct.persFig([ax, bx, cx], xlab='x [pixels]', ylab='y [pixels]')

    #         plt.show(block=False)

    #     def move(disp):
    #         a, b = disp[0], disp[1]
    #         print(a, b)
    #         if a >= 0:
    #             laa, raa, lab, rab = a, nx, 0, nx - a
    #         else:
    #             a = -a
    #             laa, raa, lab, rab = 0, nx - a, a, nx

    #         if b >= 0:
    #             lba, rba, lbb, rbb = b, ny, 0, ny - b
    #         else:
    #             b = -b
    #             lba, rba, lbb, rbb = 0, ny - b, b, ny

    #         alpha_patch = lZone[lba: rba, laa: raa]
    #         beta_patch = rZone[lbb: rbb, lab: rab]

    #         diff = alpha_patch - beta_patch

    #         if bplt:
    #             plot_data.set_data(diff)

    #             rectb.set_xy((laa, lba))
    #             rectb.set_width(raa - laa)
    #             rectb.set_height(rba - lba)
    #             bx.add_patch(rectb)
    #             rectc.set_xy((lab, lbb))
    #             rectc.set_width(rab - lab)
    #             rectc.set_height(rbb - lbb)

    #             fig.canvas.draw()
    #             plt.pause(0.05)

    #         return np.linalg.norm(diff)  # maybe an ssd (sum of square difference with a gaussian kernel is better)

    #     # a and b are ints since they rapresent pixel translations
    #     nPixelMaxDisp = pixelScan if not bplt else 20
    #     bestTranslation = optimize.brute(
    #         move,
    #         ranges=((slice(-nPixelMaxDisp, nPixelMaxDisp, 1),) * 2),
    #         disp=True,
    #         finish=None
    #     )

    #     print(bestTranslation)
