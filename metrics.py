import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
from inside_mesh.triangle_hash import TriangleHash as _TriangleHash

'''

Some code included from 'inside_mesh' library of Occupancy Networks
https://github.com/autonomousvision/occupancy_networks

'''


def define_grid_3d(N, voxel_origin=[-1, -1, -1], voxel_size=None):
    ''' define NxNxN coordinate grid across [-1, 1]
        voxel_origin is the (bottom, left, down) corner, not the middle '''

    if not voxel_size:
        voxel_size = 2.0 / (N - 1)

    # initialize empty tensors
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    grid = torch.zeros(N ** 3, 3)

    # transform first 3 columns to be x, y, z voxel index
    # every possible comb'n of [0..N,0..N,0..N]
    grid[:, 2] = overall_index % N  # [0,1,2,...,N-1,N,0,1,2,...,N]
    grid[:, 1] = (overall_index.long() // N) % N  # [N [N 0's, ..., N N's]]
    grid[:, 0] = ((overall_index.long() // N) // N) % N  # [N*N 0's,...,N*N N's]

    # transform first 3 columns: voxel indices --> voxel coordinates
    grid[:, 0] = (grid[:, 0] * voxel_size) + voxel_origin[2]
    grid[:, 1] = (grid[:, 1] * voxel_size) + voxel_origin[1]
    grid[:, 2] = (grid[:, 2] * voxel_size) + voxel_origin[0]

    return grid


def compute_iou(path_gt, path_pr, N=128, sphere=False, sphere_radius=0.25):
    ''' compute iou score
        parameters
            path_gt: path to ground-truth mesh (.ply or .obj)
            path_pr: path to predicted mesh (.ply or .obj)
            N: NxNxN grid resolution at which to compute iou '''

    # define NxNxN coordinate grid across [-1,1]
    grid = np.array(define_grid_3d(N))

    # load mesh
    occ_pr = MeshDataset(path_pr)

    # compute occupancy at specified grid points
    if sphere:
        occ_gt = torch.from_numpy(np.linalg.norm(grid, axis=-1) <= sphere_radius)
    else:
        occ_gt = MeshDataset(path_gt)
        occ_gt = torch.tensor(check_mesh_contains(occ_gt.mesh, grid))

    occ_pr = torch.tensor(check_mesh_contains(occ_pr.mesh, grid))

    # compute iou
    area_union = torch.sum((occ_gt | occ_pr).float())
    area_intersect = torch.sum((occ_gt & occ_pr).float())
    iou = area_intersect / area_union

    return iou.item()


def compute_trimesh_chamfer(mesh1, mesh2, num_mesh_samples=300000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    gen_points_sampled = trimesh.sample.sample_surface(mesh1, num_mesh_samples)[0]
    gt_points_np = trimesh.sample.sample_surface(mesh2, num_mesh_samples)[0]

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    chamfer_dist = gt_to_gen_chamfer + gen_to_gt_chamfer
    return chamfer_dist


class MeshDataset():
    def __init__(self, path_mesh, sample=False, num_pts=0):

        if not path_mesh:
            return

        self.mesh = trimesh.load(path_mesh, process=False,
                                 force='mesh', skip_materials=True)


def check_mesh_contains(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains


class MeshIntersector:
    def __init__(self, mesh, resolution=512):
        triangles = mesh.vertices[mesh.faces].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)

        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min
        self._triangles = triangles = self.rescale(triangles)

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=np.bool)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])

        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)

        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=np.bool)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains
