import math
import os
import errno
import matplotlib.colors as colors
import skimage
import skimage.filters
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import urllib.request
from tqdm import tqdm
import numpy as np
import copy
import trimesh
from inside_mesh import inside_mesh

from scipy.spatial import cKDTree as spKDTree
from data_structs import QuadTree, OctTree


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Dataset: loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Dataset: finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class OccupancyDataset():
    def __init__(self, pc_or_mesh_filename):
        self.intersector = None
        self.kd_tree = None
        self.kd_tree_sp = None
        self.mode = None

        if not pc_or_mesh_filename:
            return

        print("Dataset: loading mesh")
        self.mesh = trimesh.load(pc_or_mesh_filename, process=False, force='mesh', skip_materials=True)

        def normalize_mesh(mesh):
            print("Dataset: scaling parameters: ", mesh.bounding_box.extents)
            mesh.vertices -= mesh.bounding_box.centroid
            mesh.vertices /= np.max(mesh.bounding_box.extents / 2)

        normalize_mesh(self.mesh)

        self.intersector = inside_mesh.MeshIntersector(self.mesh, 2048)
        self.mode = 'volume'

        print('Dataset: sampling points on mesh')
        samples = trimesh.sample.sample_surface(self.mesh, 20000000)[0]

        self.kd_tree_sp = spKDTree(samples)

    def __len__(self):
        return 1

    def evaluate_occupancy(self, pts):
        return self.intersector.query(pts).astype(int).reshape(-1, 1)


class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img


class ImageFile(Dataset):
    def __init__(self, filename, url=None, grayscale=True):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 1000000000
        file_exists = os.path.isfile(filename)

        if not file_exists:
            if url is None:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename)
            else:
                print('Downloading image file...')
                urllib.request.urlretrieve(url, filename)

        self.img = Image.open(filename)
        if grayscale:
            self.img = self.img.convert('L')

        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class Patch2DWrapperMultiscaleAdaptive(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_size=(16, 16), sidelength=None, random_coords=False,
                 jitter=True, num_workers=0, length=1000, scale_init=3, max_patches=1024):

        self.length = length
        if len(sidelength) == 1:
            sidelength = 2*sidelength
        self.sidelength = sidelength

        for i in range(2):
            assert float(sidelength[i]) / float(patch_size[i]) % 1 == 0, 'Resolution not divisible by patch size'
        assert float(sidelength[0]) / float(patch_size[0]) == float(sidelength[1]) / float(patch_size[1]), \
            'number of patches must be same along each dim; check values of resolution and patch size'

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        # initialize quad tree
        self.quadtree = QuadTree(sidelength, patch_size)
        self.num_scales = self.quadtree.max_quadtree_level - self.quadtree.min_quadtree_level + 1
        self.max_patches = max_patches

        # set patches at coarsest level to be active
        patches = self.quadtree.get_patches_at_level(scale_init)
        for p in patches:
            p.activate()

        # handle parallelization
        self.num_workers = num_workers

        # make a copy of the tree for each worker
        self.quadtrees = []
        print('Dataset: preparing dataloaders...')
        for idx in tqdm(range(num_workers)):
            self.quadtrees.append(copy.deepcopy(self.quadtree))
        self.last_active_patches = self.quadtree.get_active_patches()

        # set random patches to be active
        # self.quadtree.activate_random()

        self.patch_size = patch_size
        self.dataset = dataset
        self.img = self.transform(self.dataset[0])
        self.jitter = jitter
        self.eval = False

    def toggle_eval(self):
        if not self.eval:
            self.jitter_bak = self.jitter
            self.jitter = False
            self.eval = True
        else:
            self.jitter = self.jitter_bak
            self.eval = False

    def interpolate_bilinear(self, img, fine_abs_coords, psize):
        n_blocks = fine_abs_coords.shape[0]
        n_channels = img.shape[0]
        fine_abs_coords = fine_abs_coords.reshape(n_blocks, psize[0], psize[1], 2)
        x = fine_abs_coords[..., :1]
        y = fine_abs_coords[..., 1:]
        coords = torch.cat([y, x], dim=-1)

        out = []
        for block in coords:
            tmp = torch.nn.functional.grid_sample(img[None, ...], block[None, ...],
                                                  mode='bilinear',
                                                  padding_mode='reflection',
                                                  align_corners=False)
            out.append(tmp)
        out = torch.cat(out, dim=0)
        out = out.permute(0, 2, 3, 1)
        return out.reshape(n_blocks, np.prod(psize), n_channels)

    def synchronize(self):
        self.last_active_patches = self.quadtree.get_active_patches()

        if self.num_workers == 0:
            return
        else:
            for idx in range(self.num_workers):
                self.quadtrees[idx].synchronize(self.quadtree)

    def __len__(self):
        # return len(self.dataset)
        return self.length

    def get_frozen_patches(self):
        quadtree = self.quadtree

        # get fine coords, get frozen patches is only called at eval
        fine_rel_coords, fine_abs_coords, vals,\
            coord_patch_idx = quadtree.get_frozen_samples()

        return fine_abs_coords, vals

    def __getitem__(self, idx):

        quadtree = self.quadtree
        if not self.eval and self.num_workers > 0:
            worker_idx = torch.utils.data.get_worker_info().id
            quadtree = self.quadtrees[worker_idx]

        # get fine coords
        fine_rel_coords, fine_abs_coords, coord_patch_idx = quadtree.get_stratified_samples(self.jitter, eval=self.eval)

        # get block coords
        patches = quadtree.get_active_patches()
        coords = torch.stack([p.block_coord for p in patches], dim=0)
        scales = torch.stack([torch.tensor(p.scale) for p in patches], dim=0)[:, None]
        scales = 2*scales // (self.num_scales-1) - 1
        coords = torch.cat((coords, scales), dim=-1)

        if self.eval:
            coords = coords[coord_patch_idx]

        fine_abs_coords = fine_abs_coords
        img = self.interpolate_bilinear(self.img, fine_abs_coords, self.patch_size)

        in_dict = {'coords': coords,
                   'fine_abs_coords': fine_abs_coords,
                   'fine_rel_coords': fine_rel_coords}
        gt_dict = {'img': img}

        return in_dict, gt_dict

    def update_patch_err(self, err_per_patch, step):
        assert err_per_patch.shape[0] == len(self.last_active_patches), \
            f"Trying to update the error in active patches but list of patches and error tensor" \
            f" sizes are mismatched: {err_per_patch.shape[0]} vs {len(self.last_active_patches)}"

        for i, p in enumerate(self.last_active_patches):
            # Log the history of error
            p.update_error(err_per_patch[i], step)

    def update_tiling(self):
        return self.quadtree.solve_optim(self.max_patches)


class Block3DWrapperMultiscaleAdaptive(torch.utils.data.Dataset):
    def __init__(self, dataset, octant_size=16, sidelength=None, random_coords=False,
                 max_octants=600, jitter=True, num_workers=0, length=1000, scale_init=3):

        self.length = length
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength, sidelength)
        self.sidelength = sidelength

        # initialize quad tree
        self.octtree = OctTree(sidelength, octant_size, mesh_kd_tree=dataset.kd_tree_sp)
        self.num_scales = self.octtree.max_octtree_level - self.octtree.min_octtree_level + 1

        # set patches at coarsest level to be active
        octants = self.octtree.get_octants_at_level(scale_init)
        for p in octants:
            p.activate()

        # handle parallelization
        self.num_workers = num_workers

        # make a copy of the tree for each worker
        self.octtrees = []
        print('Dataset: preparing dataloaders...')
        for idx in tqdm(range(num_workers)):
            self.octtrees.append(copy.deepcopy(self.octtree))
        self.last_active_octants = self.octtree.get_active_octants()

        self.octant_size = octant_size
        self.dataset = dataset
        self.pointcloud = None
        self.jitter = jitter
        self.eval = False

        self.max_octants = max_octants

        self.iter = 0

    def toggle_eval(self):
        if not self.eval:
            self.jitter_bak = self.jitter
            self.jitter = False
            self.eval = True
        else:
            self.jitter = self.jitter_bak
            self.eval = False

    def synchronize(self):
        self.last_active_octants = self.octtree.get_active_octants()
        if self.num_workers == 0:
            return
        else:
            for idx in range(self.num_workers):
                self.octtrees[idx].synchronize(self.octtree)

    def __len__(self):
        return self.length

    def get_frozen_octants(self, oversample):
        octtree = self.octtree

        # get fine coords, get frozen patches is only called at eval
        fine_rel_coords, fine_abs_coords, vals,\
            coord_patch_idx = octtree.get_frozen_samples(oversample)

        return fine_abs_coords, vals

    def get_eval_samples(self, oversample):
        octtree = self.octtree

        # get fine coords
        fine_rel_coords, fine_abs_coords, coord_octant_idx, _ = octtree.get_stratified_samples(self.jitter, eval=True, oversample=oversample)

        # get block coords
        octants = octtree.get_active_octants()
        coords = torch.stack([p.block_coord for p in octants], dim=0)
        scales = torch.stack([torch.tensor(p.scale) for p in octants], dim=0)[:, None]
        scales = 2*scales / (self.num_scales-1) - 1
        coords = torch.cat((coords, scales), dim=-1)

        coords = coords[coord_octant_idx]

        # query for occupancy
        sz_b, sz_p, _ = fine_abs_coords.shape

        in_dict = {'coords': coords,
                   'fine_abs_coords': fine_abs_coords,
                   'fine_rel_coords': fine_rel_coords,
                   'coord_octant_idx': torch.tensor(coord_octant_idx, dtype=torch.int)}

        return in_dict

    def __getitem__(self, idx):
        assert(not self.eval)

        octtree = self.octtree
        if not self.eval and self.num_workers > 0:
            worker_idx = torch.utils.data.get_worker_info().id
            octtree = self.octtrees[worker_idx]

        # get fine coords
        fine_rel_coords, fine_abs_coords, coord_octant_idx, coord_global_idx = octtree.get_stratified_samples(self.jitter, eval=self.eval)

        # get block coords
        octants = octtree.get_active_octants()
        coords = torch.stack([p.block_coord for p in octants], dim=0)
        scales = torch.stack([torch.tensor(p.scale) for p in octants], dim=0)[:, None]
        scales = 2*scales / (self.num_scales-1) - 1
        coords = torch.cat((coords, scales), dim=-1)

        if self.eval:
            coords = coords[coord_octant_idx]

        # query for occupancy
        sz_b, sz_p, _ = fine_abs_coords.shape
        fine_abs_coords_query = fine_abs_coords.reshape(-1, 3).detach().cpu().numpy()

        if self.eval:
            occupancy = np.zeros(fine_abs_coords_query.shape[0])
        else:
            occupancy = self.dataset.evaluate_occupancy(fine_abs_coords_query)  # start-end/num iters
        occupancy = torch.from_numpy(occupancy).reshape(sz_b, sz_p, 1)

        self.iter += 1

        in_dict = {'coords': coords,
                   'fine_abs_coords': fine_abs_coords,
                   'fine_rel_coords': fine_rel_coords}

        if self.eval:
            in_dict.update({'coord_octant_idx': torch.tensor(coord_octant_idx, dtype=torch.int)})

        gt_dict = {'occupancy': occupancy}

        return in_dict, gt_dict

    def update_octant_err(self, err_per_octant, step):
        assert err_per_octant.shape[0] == len(self.last_active_octants), \
            f"Trying to update the error in active patches but list of patches and error tensor" \
            f" sizes are mismatched: {err_per_octant.shape[0]} vs {len(self.last_active_octants)}" \
            f"step: {step}"

        for i, p in enumerate(self.last_active_octants):
            # Log the history of error
            p.update_error(err_per_octant[i], step)

        self.per_octant_error = err_per_octant

    def update_tiling(self):
        return self.octtree.solve_optim(self.max_octants)
