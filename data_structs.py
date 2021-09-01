import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches


class QuadTree():
    def __init__(self, sidelength, patch_size):
        self.sidelength = sidelength

        self.patch_size = patch_size
        self.min_patch_size = np.min(patch_size)
        self.max_patch_size = np.min(sidelength)
        self.aspect_ratio = np.array(patch_size) / np.min(patch_size)

        # how many levels of quadtree are there
        self.min_quadtree_level = int(np.log2(np.min(self.sidelength) // self.max_patch_size))
        self.max_quadtree_level = int(np.log2(np.min(sidelength) // self.min_patch_size))
        self.num_scales = self.max_quadtree_level - self.min_quadtree_level + 1

        # optimization model
        self.optim_model = gp.Model()
        self.optim_model.setParam('OutputFlag', 0)
        self.c_max_patches = None

        # initialize tree
        self.root = self.init_root(self.max_quadtree_level)

        # populate tree nodes with coordinate values, metadata
        self.populate_tree()

    def __deepcopy__(self, memo):
        deep_copied_obj = QuadTree(self.sidelength, self.patch_size)

        for k, v in self.__dict__.items():
            if k in ['optim_model', 'c_max_patches']:
                # setattr(deep_copied_obj, k, v)
                del(deep_copied_obj.__dict__[k])
            else:
                setattr(deep_copied_obj, k, copy.deepcopy(v, memo))
        return deep_copied_obj

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if k in ['optim_model', 'c_max_patches']:
                del(state[k])
        return state

    def __load__(self, obj):
        for k, v in obj.__dict__.items():
            if k == 'root':
                continue
            setattr(self, k, v)
        self.root = self.init_root(obj.max_quadtree_level)
        self.populate_tree()

        def _load_helper(curr_patch, curr_obj_patch):
            curr_patch.__load__(curr_obj_patch)
            for child, obj_child in zip(curr_patch.children, curr_obj_patch.children):
                _load_helper(child, obj_child)
            return
        _load_helper(self.root, obj.root)

    def __str__(self, level=0):

        def _str_helper(curr_patch, level):
            ret = "\t"*level+repr(curr_patch.active)+"\n"
            for child in curr_patch.children:
                ret += _str_helper(child, level+1)
            return ret
        return _str_helper(self.root, 0)

    def populate_tree(self):

        # get block coords for patches at each scale
        patch_sizes = []
        curr_size = self.max_patch_size
        while True:
            patch_sizes.append(curr_size)
            curr_size //= 2
            if curr_size == self.min_patch_size:
                patch_sizes.append(curr_size)
                break
            elif curr_size < self.min_patch_size:
                raise ValueError('Patch sizes and resolution are incompatible')

        block_coords = [self.get_block_coords(patch_size=patch_size, include_ends=True) for patch_size in patch_sizes]
        block_sizes = [block[1, 1, :] - block[0, 0, :] for block in block_coords]
        block_coords = [block[:-1, :-1, :] for block in block_coords]

        # create sampling grids for training
        num_samples = self.min_patch_size * self.aspect_ratio
        row_posts = torch.linspace(-1, 1, int(self.min_patch_size*self.aspect_ratio[0])+1)[:-1]
        col_posts = torch.linspace(-1, 1, int(self.min_patch_size*self.aspect_ratio[1])+1)[:-1]
        row_coords, col_coords = torch.meshgrid(row_posts, col_posts)
        row_coords = row_coords.flatten()
        col_coords = col_coords.flatten()

        # create sampling grids for evaluation
        # here we need to sample every pixel within each block
        row_posts = [torch.linspace(-1, 1, int(pixel_size*self.aspect_ratio[0])+1)[:-1] for pixel_size in patch_sizes]
        col_posts = [torch.linspace(-1, 1, int(pixel_size*self.aspect_ratio[1])+1)[:-1] for pixel_size in patch_sizes]
        eval_coords = [torch.meshgrid(row_post, col_post) for row_post, col_post in zip(row_posts, col_posts)]
        eval_row_coords = [eval_coord[0].flatten() for eval_coord in eval_coords]
        eval_col_coords = [eval_coord[1].flatten() for eval_coord in eval_coords]

        def _populate_tree_helper(patch, idx):
            # get block scale idx
            scale_idx = len(idx) - (self.min_quadtree_level)

            # do we have patches at this level?
            if scale_idx >= 0:

                # set patch parameters
                coords = block_coords[scale_idx]
                coord_idx = _index_block_coord(idx, coords.shape[0], coord=[0, 0])
                patch.block_coord = coords[coord_idx[0], coord_idx[1]]
                patch.block_size = block_sizes[scale_idx]
                patch.scale = scale_idx
                patch.pixel_size = patch_sizes[scale_idx]
                patch.num_samples = num_samples
                patch.row_coords = row_coords
                patch.col_coords = col_coords
                patch.eval_row_coords = eval_row_coords[scale_idx]
                patch.eval_col_coords = eval_col_coords[scale_idx]

            if not patch.children:
                return

            # recurse
            for i in range(4):
                child = patch.children[i]
                _populate_tree_helper(child, [*idx, i])

            return

        # given list of tree idxs in {0,1,2,3}^N, retrieve the block coordinate
        def _index_block_coord(tree_idx, length, coord=[0, 0]):
            if length == 1:
                return coord

            if tree_idx[0] == 0:
                pass
            elif tree_idx[0] == 1:
                coord[1] += length//2
            elif tree_idx[0] == 2:
                coord[0] += length//2
            elif tree_idx[0] == 3:
                coord[0] += length//2
                coord[1] += length//2
            else:
                raise ValueError("Unexpected child value, should be 0, 1, 2, or 3")

            return _index_block_coord(tree_idx[1:], length//2, coord)

        # done with setup, now actually populate the tree
        _populate_tree_helper(self.root, [])

    def init_root(self, max_level):

        def _init_root_helper(curr_patch, curr_level, max_level, optim_model):
            if curr_level == max_level:
                return
            curr_patch.children = [Patch(optim_model) for _ in range(4)]

            for patch in curr_patch.children:
                patch.parent = curr_patch
                _init_root_helper(patch, curr_level+1, max_level, optim_model)
            return

        # create root node
        root = Patch(self.optim_model)
        _init_root_helper(root, 0, max_level, self.optim_model)
        return root

    def get_block_coords(self, flatten=False, include_ends=False, patch_size=None):

        patch_size = patch_size * self.aspect_ratio

        # get size of each block
        block_size = (2 / (self.sidelength[0]-1) * patch_size[0], 2 / (self.sidelength[1]-1) * patch_size[1])

        # get block begin/end coordinates
        if include_ends:
            block_coords_y = torch.arange(-1, 1+block_size[0], block_size[0])
            block_coords_x = torch.arange(-1, 1+block_size[1], block_size[1])
        else:
            block_coords_y = torch.arange(-1, 1, block_size[0])
            block_coords_x = torch.arange(-1, 1, block_size[1])

        # repeat for every single block
        block_coords = torch.meshgrid(block_coords_y, block_coords_x)
        block_coords = torch.stack((block_coords[0], block_coords[1]), dim=-1)
        if flatten:
            block_coords = block_coords.reshape(-1, 2)

        return block_coords

    def get_patches_at_level(self, level):
        # level is the image scale: 0-> coarsest, N->finest
        if level == -1:
            level = self.max_quadtree_level

        # what quadtree level do our patches start at?
        # check input, too
        target_level = level + self.min_quadtree_level
        assert level <= (self.max_quadtree_level - self.min_quadtree_level), \
            "invalid 'level' input to get_blocks_at_level"

        def _get_patches_at_level_helper(curr_patch, curr_level, patches):
            if curr_level > target_level:
                return

            for patch in curr_patch.children:
                _get_patches_at_level_helper(patch, curr_level+1, patches)

            if curr_level == target_level:
                patches.append(curr_patch)
            return patches

        return _get_patches_at_level_helper(self.root, 0, [])

    def get_frozen_patches(self):

        def _get_frozen_patches_helper(curr_patch, patches):
            if curr_patch.frozen and curr_patch.active:
                patches.append(curr_patch)
            for patch in curr_patch.children:
                _get_frozen_patches_helper(patch, patches)
            return patches
        return _get_frozen_patches_helper(self.root, [])

    def get_active_patches(self, include_frozen_patches=False):

        def _get_active_patches_helper(curr_patch, patches):
            if curr_patch.active and \
                (include_frozen_patches or
                    (not include_frozen_patches and not curr_patch.frozen)):
                patches.append(curr_patch)
            for patch in curr_patch.children:
                _get_active_patches_helper(patch, patches)
            return patches
        return _get_active_patches_helper(self.root, [])

    def activate_random(self):
        def _activate_random_helper(curr_patch):
            if not curr_patch.children:
                curr_patch.activate()
                return
            elif (curr_patch.scale is not None) and (torch.rand(1).item() < 0.2):
                curr_patch.activate()
                return

            for patch in curr_patch.children:
                _activate_random_helper(patch)
            return

        _activate_random_helper(self.root)

    def synchronize(self, master):
        # set active/inactive nodes to be the same as master
        # for now just toggle the flags without worrying about the gurobi variables
        def _synchronize_helper(curr_patch, curr_patch_master):
            curr_patch.active = curr_patch_master.active
            if not curr_patch.children:
                return

            for patch, patch_master in zip(curr_patch.children, curr_patch_master.children):
                _synchronize_helper(patch, patch_master)
            return

        _synchronize_helper(self.root, master.root)

    def get_frozen_samples(self):
        patches = self.get_frozen_patches()
        if not patches:
            return None, None, None, None

        rel_coords, abs_coords, vals = [], [], []
        patch_idx = []

        for idx, p in enumerate(patches):
            rel_samp, abs_samp = p.get_stratified_samples(jitter=False, eval=True)

            rel_samp = rel_samp.reshape(-1, int(np.prod(self.min_patch_size * self.aspect_ratio)), 2)
            abs_samp = abs_samp.reshape(-1, int(np.prod(self.min_patch_size * self.aspect_ratio)), 2)
            patch_idx.extend(rel_samp.shape[0] * [idx, ])

            rel_coords.append(rel_samp)
            abs_coords.append(abs_samp)
            # values have the same size as rel_samp but last dim is a scalar
            vals.append(p.value*torch.ones(abs_samp.shape[:-1] + (1,)))

        return torch.cat(rel_coords, dim=0), torch.cat(abs_coords, dim=0), \
            torch.cat(vals, dim=0), patch_idx

    def get_stratified_samples(self, jitter=True, eval=False):
        patches = self.get_active_patches()

        rel_coords, abs_coords = [], []
        patch_idx = []

        for idx, p in enumerate(patches):
            rel_samp, abs_samp = p.get_stratified_samples(jitter=jitter, eval=eval)

            # always batch the coordinates in groups of a specific patch size
            # so we can process them in parallel
            rel_samp = rel_samp.reshape(-1, int(np.prod(self.min_patch_size * self.aspect_ratio)), 2)
            abs_samp = abs_samp.reshape(-1, int(np.prod(self.min_patch_size * self.aspect_ratio)), 2)

            # since patch samples could be split across batches,
            # keep track of which batch idx maps to which patch idx
            patch_idx.extend(rel_samp.shape[0] * [idx, ])

            rel_coords.append(rel_samp)
            abs_coords.append(abs_samp)
        return torch.cat(rel_coords, dim=0), torch.cat(abs_coords, dim=0), patch_idx

    def solve_optim(self, max_num_patches=1024):
        patches = self.get_active_patches()

        assert (len(patches) <= max_num_patches), \
            "You are trying to solve a model which is infeasible: " \
            "Number of active patches > Max number of patches"

        if self.c_max_patches is not None:
            self.optim_model.remove(self.c_max_patches)

        # global "knapsack" constraint
        expr_sum_patches = [p.update_merge() for p in patches]
        self.c_max_patches = self.optim_model.addConstr(gp.quicksum(expr_sum_patches) <= max_num_patches)

        # objective
        self.optim_model.setObjective(gp.quicksum([p.get_cost() for p in patches]), GRB.MINIMIZE)
        self.optim_model.optimize()
        obj_val = self.optim_model.objVal

        if self.optim_model.Status == GRB.INFEASIBLE:
            print("----------- Model is infeasible")
            self.optim_model.computeIIS()
            self.optim_model.write("model.ilp")

        # split and merge
        merged = 0
        split = 0
        none = 0
        for p in patches:
            # print(p)
            if p.has_split() and p.scale < self.max_quadtree_level:
                p.deactivate()
                for child in p.get_children():
                    child.activate()
                split += 1
            elif p.has_merged() and p.scale >= self.min_quadtree_level and p.scale > 0:
                # we first check if it is active,
                # since we could have already been activated by a neighbor
                if p.active:
                    for neighbor in p.get_neighbors():
                        neighbor.deactivate()
                    p.parent.activate()
                merged += 1

            else:
                p.update()
                none += 1
        stats_dict = {'merged': merged,
                      'splits': split,
                      'none': none,
                      'obj': obj_val}
        print(f"============================= Total patches:{len(patches)}, split/merge:{split}/{merged}")
        return stats_dict

    def draw(self):
        fig, ax = plt.subplots(1, figsize=(5, 5))
        depth = 1 + self.max_quadtree_level - self.min_quadtree_level
        sidelen = 4**(depth-1) // 2**(depth-1)

        # calculate scale
        patch_list = self.get_active_patches()
        patches_err = [p.err for p in patch_list]

        max_err = np.max(patches_err)
        min_err = np.min(patches_err)

        cmap = plt.cm.get_cmap('viridis')

        def _draw_level(patch, curr_level, ax, sidelen, offset, scale):
            if curr_level > self.max_quadtree_level:
                return ax

            scale = scale/2.

            for i, child in enumerate(patch.children):
                if i == 0:
                    new_offset = (offset[0], offset[1])
                elif i == 1:
                    new_offset = (offset[0] + scale * sidelen, offset[1])
                elif i == 2:
                    new_offset = (offset[0], offset[1] + scale * sidelen)
                else:
                    new_offset = (offset[0] + scale * sidelen, offset[1] + scale * sidelen)

                if child.active:
                    norm_err = (child.err-min_err)/(max_err-min_err)
                    if child.frozen:
                        facecolor = 'white'
                        edgecolor = 'red'
                    else:
                        facecolor = cmap(norm_err)
                        edgecolor = 'black'
                    rect = patches.Rectangle(new_offset, scale * sidelen, scale * sidelen, linewidth=1,
                                             edgecolor=edgecolor,
                                             facecolor=facecolor, fill=True)
                    ax.add_patch(rect)
                else:
                    ax = _draw_level(child, curr_level+1, ax, sidelen, new_offset, scale)

            return ax

        ax = _draw_level(self.root, self.min_quadtree_level, ax, sidelen, (0., 0.), 1.)
        ax.set_aspect('equal')
        plt.xlim(-1, sidelen + 1)
        plt.ylim(-1, sidelen + 1)
        plt.gca().invert_yaxis()  # we want 0,0 to be on top-left

        norm = colors.Normalize(vmin=min_err, vmax=max_err)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        return fig


# patch class
class Patch():
    def __init__(self, optim_model=None, block_coord=None, scale=None):
        self.active = False

        self.parent = None
        self.children = []

        # absolute block coordinate
        self.block_coord = block_coord

        # size of block in absolute coord frame
        self.block_size = None

        # scale level of block
        self.scale = scale

        # num samples to be generated for this block
        self.num_samples = None

        # num pixels in this patch
        self.pixel_size = None

        # optimization model
        self.optim = optim_model

        # row/column coords for sampling at test time
        # initialized by set_samples() function
        self.row_coords = None
        self.col_coords = None
        self.eval_row_coords = None
        self.eval_col_coords = None

        # error for doing nothing, merging, splitting
        self.err = 0.
        self.last_updated = 0.

        self._nocopy = ['optim', 'I_grp', 'I_split', 'I_none',
                        'I_merge', 'c_joinable', 'c_merge_split']
        self._pickle_vars = ['parent', 'children', 'active', 'err', 'last_updated']
        self.spec_cstrs = []

        # options for pruning
        self.frozen = False
        self.value = 0.0

    def __str__(self):
        str = f"Patch id={id(self)}\n" \
              f" . active={self.active}\n" \
              f" . level={self.scale}\n" \
              f" . model={self.optim}"

        if self.active:
            str += f"\n . g={self.I_grp.x}, s={self.I_split.x}, n={self.I_none.x}"

        return str

    # override deep copy to copy undeepcopyable objects by reference
    def __deepcopy__(self, memo):
        deep_copied_obj = Patch()
        for k, v in self.__dict__.items():
            if k in self._nocopy:
                # setattr(deep_copied_obj, k, None)
                if k in deep_copied_obj.__dict__.keys():
                    del(deep_copied_obj.__dict__[k])
            else:
                setattr(deep_copied_obj, k, copy.deepcopy(v, memo))

        return deep_copied_obj

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if k in self._nocopy:
            # if k not in self._pickle_vars:
                del(state[k])
        return state

    def __load__(self, obj):
        for k, v in obj.__dict__.items():
            if k in ['children', 'parent']:
                continue
            setattr(self, k, v)
        if self.active:
            self.activate()

    def update(self):
        self.deactivate()
        self.activate()

    def activate(self):
        self.active = True

        # indicator variables
        self.I_grp = self.optim.addVar(vtype=GRB.BINARY)
        self.I_split = self.optim.addVar(vtype=GRB.BINARY)
        self.I_none = self.optim.addVar(vtype=GRB.BINARY)

        self.I_merge = gp.LinExpr(0.0)

        # local constraint "merge/none/split"
        self.c_joinable = self.optim.addConstr(self.I_grp + self.I_none + self.I_split == 1)

        # local constraint "merge-split"
        self.c_merge_split = None

    def deactivate(self):
        self.active = False

        self.optim.remove(self.I_grp)
        self.optim.remove(self.I_split)
        self.optim.remove(self.I_none)

        self.I_merge = gp.LinExpr(0.0)

        self.optim.remove(self.c_joinable)

        if self.c_merge_split is not None:
            self.optim.remove(self.c_merge_split)

        for cstr in self.spec_cstrs:
            self.optim.remove(cstr)
        self.spec_cstrs = []

    def is_mergeable(self):
        siblings = self.parent.children
        return np.all(np.all([sib.active for sib in siblings]))

    def set_sample_params(self, num_samples):
        self.num_samples = num_samples
        posts = torch.linspace(-1, 1, self.num_samples+1)[:-1]
        row_coords, col_coords = torch.meshgrid(posts, posts)
        self.row_coords = row_coords.flatten()
        self.col_coords = col_coords.flatten()

    def must_split(self):
        self.spec_cstrs.append(
            self.optim.addConstr(self.I_split == 1)
        )

    def must_merge(self):
        self.spec_cstrs.append(
            self.optim.addConstr(self.I_grp == 1)
        )

    def has_split(self):
        return self.I_split.x == 1

    def has_merged(self):
        return self.I_grp.x == 1
        # return self.I_none.x==0 and self.I_split.x==0

    def has_done_nothing(self):
        return self.I_none.x == 1

    def get_cost(self):
        area = self.block_size[0]**2
        alpha = 0.2  # how much worse we expect the error to be when merging
        beta = -0.02  # how much better we expect the error to be when splitting

        # == Merge
        if self.scale > 0:  # it should never be root, but still..
            err_merge = (4+alpha) * area * self.err

            if self.parent.last_updated:
                parent_area = self.parent.block_size[0]**2
                err_merge = parent_area * self.parent.err
        else:
            err_merge = self.err

        # == Split
        if self.children:
            err_split = (0.25+beta) * area * self.err

            if self.children[0].last_updated:
                err_children = np.sum([child.err for child in self.children])
                err_split = area * err_children
        else:
            err_split = 1.  # in case you don't have children, high to avoid splitting

        # == None
        err_none = area * self.err

        return err_none * self.I_none \
            + err_split * self.I_split \
            + err_merge * self.I_grp

    def update_merge(self):
        if self.parent is None:  # if root
            return gp.LinExpr(0)

        siblings = self.parent.children
        if np.all([sib.active for sib in siblings]):
            I_grp_neighs = [s.I_grp for s in siblings]
            self.I_merge = gp.quicksum(I_grp_neighs)

        # local constraint "joinable"
        self.c_merge_split = self.optim.addConstr(self.I_none + self.I_split + .25*self.I_merge == 1)
        expr_max_patches = 4 * self.I_split + 1 * self.I_none + .25 * self.I_grp

        return expr_max_patches

    def get_neighbors(self):
        return self.parent.children

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def is_joinable(self):
        # test if siblings are all leaf nodes
        siblings = self.parent.children
        return np.all([sib.active for sib in siblings])

    def get_block_coord(self):
        return self.block_coord

    def get_scale(self):
        return self.scale

    def update_error(self, error, iter):
        self.err = error
        self.last_updated = iter

    def get_stratified_samples(self, jitter=True, eval=False):
        # Block coords are always aligned to the pixel grid,
        # e.g., they align with pixels 0, 8, 16, 24, etc. for
        # patch size 8
        #
        # To normalize the coordinates between (-1, 1), consider
        # we have an image of 64x64 and patch size 8x8.
        # The block coordinate (-1, -1) aligns with pixel (0, 0)
        # and coordinate (1, 1) aligns with pixel (63, 63)
        #
        # Absolute coordinates within a block should stretch all the way
        # from the absolute position of one block coordinate to another.
        # Say each block contains 8x8 pixels and we use a feature grid
        # of 8x8 features to interpolate values within a block.
        # This means is that the feature positions are not actually
        # aligned to the pixel positions. The features are positioned
        # on a grid stretching from one block coord to another whereas
        # the pixel grid ends just short of the next block coordinate
        #
        # Example patch (x = pixel position, B = block coordinate position)
        # and relative coordinate positions.
        #
        # -1 ^ B x x x x x x x B
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #  1 v B x x x x x x x B
        #      <--------------->
        #     -1               1
        #
        # When we generate samples for a patch, we sample an
        # 8x8 grid that extends between block coords, i.e.
        # between the arrows above
        #
        if eval:
            row_coords = self.eval_row_coords.flatten()
            col_coords = self.eval_col_coords.flatten()
        else:
            row_coords = self.row_coords
            col_coords = self.col_coords

            if jitter:
                row_coords = self.row_coords + torch.rand_like(self.row_coords) * 2./self.num_samples[0]
                col_coords = self.col_coords + torch.rand_like(self.col_coords) * 2./self.num_samples[1]

        rel_samples = torch.stack((row_coords, col_coords), dim=-1)
        abs_samples = self.block_coord[None, :] + self.block_size[None, :] * (rel_samples+1)/2
        return rel_samples, abs_samples


class OctTree():
    def __init__(self, sidelength, min_octant_size, bounds=((-1, 1), (-1, 1), (-1, 1)), mesh_kd_tree=None):
        self.sidelength = sidelength
        self.min_octant_size = min_octant_size
        self.max_octant_size = sidelength[0]

        # how many levels of quadtree are there
        self.min_octtree_level = int(np.log2(self.sidelength[0] // self.max_octant_size))
        self.max_octtree_level = int(np.log2(sidelength[0] // min_octant_size))
        self.num_scales = self.max_octtree_level - self.min_octtree_level + 1

        # optimization model
        self.optim_model = gp.Model()
        self.optim_model.setParam('OutputFlag', 0)
        self.c_max_octants = None

        # set bounds
        self.z_min, self.z_max = bounds[0]
        self.y_min, self.y_max = bounds[1]
        self.x_min, self.x_max = bounds[2]

        # KD tree that stores points on mesh surface
        self.surface_tree = mesh_kd_tree

        # initialize tree
        self.root = self.init_root(self.max_octtree_level)

        # populate tree nodes with coordinate values, metadata
        self.populate_tree()

    def __deepcopy__(self, memo):
        deep_copied_obj = OctTree(self.sidelength, self.min_octant_size)

        for k, v in self.__dict__.items():
            if k in ['optim_model', 'c_max_octants']:
                setattr(deep_copied_obj, k, v)
            else:
                setattr(deep_copied_obj, k, copy.deepcopy(v, memo))
        return deep_copied_obj

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if k in ['optim_model', 'c_max_octants']:
                del(state[k])
        return state

    def __load__(self, obj):
        for k, v in obj.__dict__.items():
            if k == 'root':
                continue
            setattr(self, k, v)
        self.root = self.init_root(obj.max_octtree_level)
        self.populate_tree()

        def _load_helper(curr_patch, curr_obj_patch):
            curr_patch.__load__(curr_obj_patch)
            for child, obj_child in zip(curr_patch.children, curr_obj_patch.children):
                _load_helper(child, obj_child)
            return
        _load_helper(self.root, obj.root)

    def __str__(self, level=0):

        def _str_helper(curr_octant, level):
            ret = "\t"*level+repr(curr_octant.active)+"\n"
            for child in curr_octant.children:
                ret += _str_helper(child, level+1)
            return ret
        return _str_helper(self.root, 0)

    def populate_tree(self):

        # maximum octant scale
        max_octant_scale = int(np.log2(self.max_octant_size))
        min_octant_scale = int(np.log2(self.min_octant_size))

        # get block coords for octants at each scale
        octant_sizes = [2**s for s in range(min_octant_scale, max_octant_scale+1)]
        octant_sizes.reverse()
        block_coords = [self.get_block_coords(octant_size=octant_size, include_ends=True) for octant_size in octant_sizes]
        block_sizes = [block[1, 1, 1, :] - block[0, 0, 0, :] for block in block_coords]
        block_coords = [block[:-1, :-1, :-1, :] for block in block_coords]

        # create sampling grids for training
        num_samples = self.min_octant_size
        posts = torch.linspace(-1, 1, self.min_octant_size+1)[:-1]
        row_coords, col_coords, dep_coords = torch.meshgrid(posts, posts, posts)
        row_coords = row_coords.flatten()
        col_coords = col_coords.flatten()
        dep_coords = dep_coords.flatten()

        # create sampling grids for evaluation
        # here we need to sample every voxel within each block
        posts = [torch.linspace(-1, 1, voxel_size+1)[:-1] + (1/voxel_size)/2 for voxel_size in octant_sizes]
        eval_coords = [torch.meshgrid(post, post, post) for post in posts]
        eval_row_coords = [eval_coord[0].flatten() for eval_coord in eval_coords]
        eval_col_coords = [eval_coord[1].flatten() for eval_coord in eval_coords]
        eval_dep_coords = [eval_coord[2].flatten() for eval_coord in eval_coords]

        def _populate_tree_helper(octant, idx):
            # get block scale idx
            scale_idx = len(idx) - (self.min_octtree_level)

            # do we have octants at this level?
            if scale_idx >= 0:

                # set patch parameters
                coords = block_coords[scale_idx]
                coord_idx = _index_block_coord(idx, coords.shape[0], coord=[0, 0, 0])
                octant.block_coord = coords[coord_idx[0], coord_idx[1], coord_idx[2]]
                octant.block_size = block_sizes[scale_idx]
                octant.scale = scale_idx
                octant.voxel_size = octant_sizes[scale_idx]
                octant.num_samples = num_samples
                octant.row_coords = row_coords
                octant.col_coords = col_coords
                octant.dep_coords = dep_coords
                octant.eval_row_coords = eval_row_coords[scale_idx]
                octant.eval_col_coords = eval_col_coords[scale_idx]
                octant.eval_dep_coords = eval_dep_coords[scale_idx]
                octant.surface_tree = self.surface_tree

            if not octant.children:
                return

            # recurse
            for i in range(8):
                child = octant.children[i]
                _populate_tree_helper(child, [*idx, i])

            return

        # given list of tree idxs in {0,1,2,3}^N, retrieve the block coordinate
        def _index_block_coord(tree_idx, length, coord=[0, 0, 0]):
            if length == 1:
                return coord

            # depth 0
            if tree_idx[0] == 0:
                pass
            elif tree_idx[0] == 1:
                coord[1] += length//2
            elif tree_idx[0] == 2:
                coord[0] += length//2
            elif tree_idx[0] == 3:
                coord[0] += length//2
                coord[1] += length//2
            # depth 1
            elif tree_idx[0] == 4:
                coord[2] += length//2
            elif tree_idx[0] == 5:
                coord[1] += length//2
                coord[2] += length//2
            elif tree_idx[0] == 6:
                coord[0] += length//2
                coord[2] += length//2
            elif tree_idx[0] == 7:
                coord[0] += length//2
                coord[1] += length//2
                coord[2] += length//2
            else:
                raise ValueError("Unexpected child value, should be in{0,...7}")

            return _index_block_coord(tree_idx[1:], length//2, coord)

        # done with setup, now actually populate the tree
        _populate_tree_helper(self.root, [])

    def init_root(self, max_level):

        def _init_root_helper(curr_octant, curr_level, max_level, optim_model):
            if curr_level == max_level:
                return
            curr_octant.children = [Octant(optim_model) for _ in range(8)]

            for octant in curr_octant.children:
                octant.parent = curr_octant
                _init_root_helper(octant, curr_level+1, max_level, optim_model)
            return

        # create root node
        root = Octant(self.optim_model)
        _init_root_helper(root, 0, max_level, self.optim_model)
        return root

    def get_block_coords(self, flatten=False, include_ends=False, octant_size=None):

        # use finest scale patch by default
        if octant_size is None:
            octant_size = self.min_octant_size  # TODO: ?? verify

        # get size of each block
        z_len = self.z_max - self.z_min
        y_len = self.y_max - self.y_min
        x_len = self.x_max - self.x_min

        block_size = (z_len / (self.sidelength[0]) * octant_size,
                      y_len / (self.sidelength[1]) * octant_size,
                      x_len / (self.sidelength[2]) * octant_size)

        # get block begin/end coordinates
        if include_ends:
            block_coords_z = torch.arange(self.z_min, self.z_max + block_size[0], block_size[0])
            block_coords_y = torch.arange(self.y_min, self.y_max + block_size[1], block_size[1])
            block_coords_x = torch.arange(self.x_min, self.x_max + block_size[2], block_size[2])
        else:
            block_coords_z = torch.arange(self.z_min, self.z_max, block_size[0])
            block_coords_y = torch.arange(self.y_min, self.y_max, block_size[1])
            block_coords_x = torch.arange(self.x_min, self.x_max, block_size[2])

        # repeat for every single block
        block_coords = torch.meshgrid(block_coords_z, block_coords_y, block_coords_x)
        block_coords = torch.stack((block_coords[0], block_coords[1], block_coords[2]), dim=-1)
        if flatten:
            block_coords = block_coords.reshape(-1, 3)

        return block_coords

    def get_octants_at_level(self, level):
        # level is the image scale: 0-> coarsest, N->finest

        # what quadtree level do our octants start at?
        # check input, too
        target_level = level + self.min_octtree_level
        assert level <= (self.max_octtree_level - self.min_octtree_level), \
            "invalid 'level' input to get_blocks_at_level"

        def _get_octants_at_level_helper(curr_octant, curr_level, octants):
            if curr_level > target_level:
                return

            for octant in curr_octant.children:
                _get_octants_at_level_helper(octant, curr_level+1, octants)

            if curr_level == target_level:
                octants.append(curr_octant)
            return octants

        return _get_octants_at_level_helper(self.root, 0, [])

    def get_frozen_octants(self):

        def _get_frozen_octants_helper(curr_octant, octants):
            if curr_octant.frozen and curr_octant.active:
                octants.append(curr_octant)
            for octant in curr_octant.children:
                _get_frozen_octants_helper(octant, octants)
            return octants
        return _get_frozen_octants_helper(self.root, [])

    def get_active_octants(self, include_frozen_octants=False):

        def _get_active_octants_helper(curr_octant, octants):
            if curr_octant.active and \
                (include_frozen_octants or
                    (not include_frozen_octants and not curr_octant.frozen)):
                octants.append(curr_octant)
                return octants
            for octant in curr_octant.children:
                _get_active_octants_helper(octant, octants)
            return octants
        return _get_active_octants_helper(self.root, [])

    def activate_random(self):
        def _activate_random_helper(curr_octant):
            if not curr_octant.children:
                curr_octant.activate()
                return
            elif (curr_octant.scale is not None) and (torch.rand(1).item() < 0.2):
                curr_octant.activate()
                return

            for patch in curr_octant.children:
                _activate_random_helper(patch)
            return

        _activate_random_helper(self.root)

    def synchronize(self, master):
        # set active/inactive nodes to be the same as master
        # for now just toggle the flags without worrying about the gurobi variables
        def _synchronize_helper(curr_octant, curr_octant_master):
            curr_octant.active = curr_octant_master.active
            curr_octant.frozen = curr_octant_master.frozen
            curr_octant.value = curr_octant_master.value
            if not curr_octant.children:
                return

            for octant, octant_master in zip(curr_octant.children, curr_octant_master.children):
                _synchronize_helper(octant, octant_master)
            return

        _synchronize_helper(self.root, master.root)

    def get_frozen_samples(self, oversample):
        octants = self.get_frozen_octants()

        if not octants:
            return None, None, None, None

        rel_coords, abs_coords, vals = [], [], []
        octant_idx = []

        for idx, p in enumerate(octants):
            rel_samp, abs_samp, _ = p.get_stratified_samples(jitter=False, eval=True, oversample=oversample)

            rel_samp = rel_samp.reshape(-1, self.min_octant_size**3, 3)
            abs_samp = abs_samp.reshape(-1, self.min_octant_size**3, 3)
            octant_idx.extend(rel_samp.shape[0] * [idx, ])

            rel_coords.append(rel_samp)
            abs_coords.append(abs_samp)
            # values have the same size as rel_samp but last dim is a scalar
            vals.append(p.value*torch.ones(abs_samp.shape[:-1] + (1,)))

        return torch.cat(rel_coords, dim=0), torch.cat(abs_coords, dim=0), \
            torch.cat(vals, dim=0), octant_idx

    def get_stratified_samples(self, jitter=True, eval=False, oversample=1):
        octants = self.get_active_octants()

        rel_coords, abs_coords = [], []
        all_global_indices = []
        octant_idx = []

        for idx, p in enumerate(octants):

            rel_samp, abs_samp, global_indices = p.get_stratified_samples(jitter=jitter, eval=eval, oversample=oversample)

            # always batch the coordinates in groups of a specific patch size
            # so we can process them in parallel
            rel_samp = rel_samp.reshape(-1, int(self.min_octant_size * oversample)**3, 3)
            abs_samp = abs_samp.reshape(-1, int(self.min_octant_size * oversample)**3, 3)
            if global_indices is not None:
                global_indices = global_indices.reshape(-1, int(self.min_octant_size * oversample)**3, 1)

            # since patch samples could be split across batches,
            # keep track of which batch idx maps to which patch idx
            octant_idx.extend(rel_samp.shape[0] * [idx, ])

            rel_coords.append(rel_samp)
            abs_coords.append(abs_samp)
            if global_indices is not None:
                all_global_indices.append(global_indices)

        return torch.cat(rel_coords, dim=0), torch.cat(abs_coords, dim=0), octant_idx, None

    def solve_optim(self, Max_Num_Octants=150):
        octants = self.get_active_octants()

        assert (len(octants) <= Max_Num_Octants), \
            "You are trying to solve a model which is infeasible: " \
            "Number of active octants > Max number of octants"

        if self.c_max_octants is not None:
            self.optim_model.remove(self.c_max_octants)

        # global "knapsack" constraint
        expr_sum_octants = [p.update_merge() for p in octants]
        self.c_max_octants = self.optim_model.addConstr(gp.quicksum(expr_sum_octants) <= Max_Num_Octants)

        # objective
        self.optim_model.setObjective(gp.quicksum([p.get_cost() for p in octants]), GRB.MINIMIZE)
        self.optim_model.optimize()
        obj_val = self.optim_model.objVal

        if self.optim_model.Status == GRB.INFEASIBLE:
            print("----------- Model is infeasible")
            self.optim_model.computeIIS()
            self.optim_model.write("model.ilp")

        # split and merge
        merged = 0
        split = 0
        none = 0
        for p in octants:
            # print(p)
            if p.has_split() and p.scale < self.max_octtree_level:
                p.deactivate()
                for child in p.get_children():
                    child.activate()
                split += 1
            elif p.has_merged() and p.scale >= self.min_octtree_level and p.scale > 0:
                # we first check if it is active,
                # since we could have already been activated by a neighbor
                if p.active:
                    for neighbor in p.get_neighbors():
                        neighbor.deactivate()
                    p.parent.activate()
                merged += 1

            else:
                p.update()
                none += 1

        stats_dict = {'merged': merged,
                      'splits': split,
                      'none': none,
                      'obj': obj_val}
        print(f"============================= Total octants:{len(octants)}, split/merge:{split}/{merged}")
        print(f"Vars={len(self.optim_model.getVars())}, Cstrs={len(self.optim_model.getConstrs())}")
        return stats_dict

    def draw(self, color_by_scale=False, save_fig=True):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca(projection='3d')

        depth = 1 + self.max_octtree_level - self.min_octtree_level
        sidelen = 4**(depth-1) // 2**(depth-1)

        # calculate scale
        octant_list = self.get_active_octants()
        octants_err = [p.err for p in octant_list]
        max_err = np.max(octants_err)
        min_err = np.min(octants_err)

        cmap = plt.cm.get_cmap('viridis')

        def cuboid_data(pos, size=(1, 1, 1)):
            eps = 0.2
            o = pos + (eps, eps, eps)
            # get the length, width, and height
            l, w, h = size
            l -= eps
            w -= eps
            h -= eps
            x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
            y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1], o[1], o[1]],
                 [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
            z = [[o[2], o[2], o[2], o[2], o[2]],
                 [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
            return np.array(x), np.array(y), np.array(z)

        def draw_cube_at(pos=(0, 0, 0), size=(1, 1, 1),
                         color='b', edgecolor='b', alpha=1., ax=None):

            # Plotting a cube element at position pos
            if ax is not None:
                Z, Y, X = cuboid_data(pos, size)
                ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=alpha,
                                edgecolors=edgecolor, linewidth=0.1)

        def _draw_level_2(octant, curr_level,
                          ax, sidelen, offset, scale):
            if curr_level > self.max_octtree_level:
                return ax

            scale = scale/2.

            for i, child in enumerate(octant.children):
                # depth 0
                if i == 0:
                    new_offset = (offset[0],
                                  offset[1],
                                  offset[2])
                elif i == 1:
                    new_offset = (offset[0] + scale * sidelen,
                                  offset[1],
                                  offset[2])
                elif i == 2:
                    new_offset = (offset[0],
                                  offset[1] + scale * sidelen,
                                  offset[2])
                elif i == 3:
                    new_offset = (offset[0] + scale * sidelen,
                                  offset[1] + scale * sidelen,
                                  offset[2])
                # depth 1
                elif i == 4:
                    new_offset = (offset[0],
                                  offset[1],
                                  offset[2] + scale * sidelen)
                elif i == 5:
                    new_offset = (offset[0] + scale * sidelen,
                                  offset[1],
                                  offset[2] + scale * sidelen)
                elif i == 6:
                    new_offset = (offset[0],
                                  offset[1] + scale * sidelen,
                                  offset[2] + scale * sidelen)
                else:
                    new_offset = (offset[0] + scale * sidelen,
                                  offset[1] + scale * sidelen,
                                  offset[2] + scale * sidelen)

                if child.active:
                    norm_err = (child.err-min_err)/(max_err-min_err)
                    sz = scale*sidelen

                    if child.frozen:
                        color = [0.5, 0.5, 0.5]
                        edgecolor = [1.0, 0.0, 0.0, 0.1]
                        alpha = 0.
                    else:
                        color = cmap(norm_err)[0:3]
                        edgecolor = 'none'
                        alpha = 0.1

                    draw_cube_at(pos=new_offset, size=(sz, sz, sz),
                                 color=color, edgecolor=edgecolor,
                                 alpha=alpha, ax=ax)

                else:
                    ax = _draw_level_2(child, curr_level+1,
                                       ax, sidelen, new_offset, scale)

            return ax

        ax = _draw_level_2(self.root, self.min_octtree_level,
                           ax, sidelen,
                           (0., 0., 0.), 1.)

        ax = fig.gca(projection='3d')
        ax.grid(False)
        ax.set_xlim(-1, sidelen+1)
        ax.set_ylim(-1, sidelen+1)
        ax.set_zlim(-1, sidelen+1)

        ax.set_xticks([-1, sidelen+1])
        ax.set_xticklabels([-1, 1])
        ax.set_yticks([-1, sidelen+1])
        ax.set_yticklabels([-1, 1])
        ax.set_zticks([-1, sidelen+1])
        ax.set_zticklabels([-1, 1])

        return fig


class Octant():
    def __init__(self, optim_model=None, block_coord=None, scale=None, gamma=0.95):
        self.active = False

        self.parent = None
        self.children = []

        # absolute block coordinate
        self.block_coord = block_coord

        # size of block in absolute coord frame
        self.block_size = None

        self.old_block_coord = None
        self.old_block_size = None

        # scale level of block
        self.scale = scale

        # num samples to be generated for this block
        self.num_samples = None

        # num pixels in this patch
        self.voxel_size = None

        # optimization model
        self.optim = optim_model

        # row/column coords for sampling at test time
        # initialized by set_samples() function
        self.row_coords = None
        self.col_coords = None
        self.dep_coords = None

        self.near_mesh_abs_samples = None
        self.near_mesh_rel_samples = None

        # error for doing nothing, merging, splitting
        self.err = 0.
        self.last_updated = 0.

        self.gamma = gamma

        self._nocopy = ['optim', 'I_grp', 'I_split', 'I_none',
                        'I_merge', 'c_joinable', 'c_merge_split',
                        'children', 'parent', 'loss', 'loss_iter',
                        'err']
        self.spec_cstrs = []

        self._pickle_vars = ['parent', 'children', 'active', 'err', 'last_updated', 'frozen', 'value']

        # options for pruning
        self.frozen = False
        self.value = 0

    def __str__(self):
        str = f"Octant id={id(self)}\n" \
              f" . active={self.active}\n" \
              f" . level={self.scale}\n" \
              f" . model={self.optim}"

        if self.active:
            str += f"\n . g={self.I_grp.x}, s={self.I_split.x}, n={self.I_none.x}"

        return str

    # override deep copy to copy undeepcopyable objects by reference
    def __deepcopy__(self, memo):
        deep_copied_obj = Octant()
        for k, v in self.__dict__.items():
            if k in self._nocopy:
                setattr(deep_copied_obj, k, v)
            else:
                setattr(deep_copied_obj, k, copy.deepcopy(v, memo))

        return deep_copied_obj

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if k not in self._pickle_vars:
                del(state[k])
        return state

    def __load__(self, obj):
        for k, v in obj.__dict__.items():
            if k in ['children', 'parent']:
                continue
            setattr(self, k, v)
        if self.active:
            self.activate()

    def update(self):
        self.deactivate()
        self.activate()

    def activate(self):
        self.active = True

        # indicator variables
        self.I_grp = self.optim.addVar(vtype=GRB.BINARY)
        self.I_split = self.optim.addVar(vtype=GRB.BINARY)
        self.I_none = self.optim.addVar(vtype=GRB.BINARY)

        self.I_merge = gp.LinExpr(0.0)

        # local constraint "merge/none/split"
        self.c_joinable = self.optim.addConstr(self.I_grp + self.I_none + self.I_split == 1)

        # local constraint "merge-split"
        self.c_merge_split = None

    def deactivate(self):
        self.active = False

        self.optim.remove(self.I_grp)
        self.optim.remove(self.I_split)
        self.optim.remove(self.I_none)

        self.I_merge = gp.LinExpr(0.0)

        self.optim.remove(self.c_joinable)

        if self.c_merge_split is not None:
            self.optim.remove(self.c_merge_split)

        for cstr in self.spec_cstrs:
            self.optim.remove(cstr)
        self.spec_cstrs = []

    def is_mergeable(self):
        siblings = self.parent.children
        return np.all(np.all([sib.active for sib in siblings]))

    def set_sample_params(self, num_samples):
        self.num_samples = num_samples
        posts = torch.linspace(-1, 1, self.num_samples+1)[:-1]
        row_coords, col_coords, dep_coords = torch.meshgrid(posts, posts, posts)
        self.row_coords = row_coords.flatten()
        self.col_coords = col_coords.flatten()
        self.dep_coords = dep_coords.flatten()

    def must_split(self):
        self.spec_cstrs.append(
            self.optim.addConstr(self.I_split == 1)
        )

    def must_merge(self):
        self.spec_cstrs.append(
            self.optim.addConstr(self.I_grp == 1)
        )

    def has_split(self):
        return self.I_split.x == 1

    def has_merged(self):
        return self.I_grp.x == 1
        # return self.I_none.x==0 and self.I_split.x==0

    def has_done_nothing(self):
        return self.I_none.x == 1

    def get_cost(self):
        area = self.block_size[0]**2
        alpha = 0.2  # how much worse we expect the error to be when merging
        beta = -0.02  # how much better we expect the error to be when splitting

        # == Merge
        if self.scale > 0:  # it should never be root, but still..
            err_merge = (8+alpha) * area * self.err

            if self.parent.last_updated:
                parent_area = self.parent.block_size[0]**2
                err_merge = parent_area * self.parent.err
        else:
            err_merge = self.err

        # == Split
        if self.children:
            err_split = (0.125+beta) * area * self.err

            if self.children[0].last_updated:
                err_children = np.sum([child.err for child in self.children])
                err_split = area * err_children
        else:
            err_split = 1.  # in case you don't have children, high to avoid splitting

        err_none = area * self.err

        return err_none * self.I_none \
            + err_split * self.I_split \
            + err_merge * self.I_grp

    def update_merge(self):
        if self.parent is None:  # if root
            return gp.LinExpr(0)

        siblings = self.parent.children
        if np.all([sib.active for sib in siblings]):
            I_grp_neighs = [s.I_grp for s in siblings]
            self.I_merge = gp.quicksum(I_grp_neighs)

        # local constraint "joinable"
        self.c_merge_split = self.optim.addConstr(self.I_none + self.I_split + .125*self.I_merge == 1)
        expr_max_patches = 8 * self.I_split + 1 * self.I_none + .125 * self.I_grp

        return expr_max_patches

    def get_neighbors(self):
        return self.parent.children

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def is_joinable(self):
        # test if siblings are all leaf nodes
        siblings = self.parent.children
        return np.all([sib.active for sib in siblings])

    def get_block_coord(self):
        return self.block_coord

    def get_scale(self):
        return self.scale

    def update_error(self, error, iter):
        self.err = self.gamma*self.err + (1-self.gamma)*error
        self.last_updated = iter

    def get_block_coords(self, flatten=False, include_ends=False, octant_size=None):
        # get size of each block
        z_len = 2
        y_len = 2
        x_len = 2

        sidelength = 256

        block_size = (z_len / (sidelength) * octant_size,
                      y_len / (sidelength) * octant_size,
                      x_len / (sidelength) * octant_size)

        # get block begin/end coordinates
        if include_ends:
            block_coords_z = torch.arange(-1, -1 + block_size[0], block_size[0])
            block_coords_y = torch.arange(-1, 1 + block_size[1], block_size[1])
            block_coords_x = torch.arange(-1, 1 + block_size[2], block_size[2])
        else:
            block_coords_z = torch.arange(-1, 1, block_size[0])
            block_coords_y = torch.arange(-1, 1, block_size[1])
            block_coords_x = torch.arange(-1, 1, block_size[2])

        # repeat for every single block
        block_coords = torch.meshgrid(block_coords_z, block_coords_y, block_coords_x)
        block_coords = torch.stack((block_coords[0], block_coords[1], block_coords[2]), dim=-1)
        if flatten:
            block_coords = block_coords.reshape(-1, 3)

        return block_coords

    def get_stratified_samples(self, jitter=True, eval=False, oversample=1., kd_tree=None):
        # Block coords are always aligned to the pixel grid,
        # e.g., they align with pixels 0, 8, 16, 24, etc. for
        # patch size 8
        #
        # To normalize the coordinates between (-1, 1), consider
        # we have an image of 64x64 and patch size 8x8.
        # The block coordinate (-1, -1) aligns with pixel (0, 0)
        # and coordinate (1, 1) aligns with pixel (63, 63)
        #
        # Absolute coordinates within a block should stretch all the way
        # from the absolute position of one block coordinate to another.
        # Say each block contains 8x8 pixels and we use a feature grid
        # of 8x8 features to interpolate values within a block.
        # This means is that the feature positions are not actually
        # aligned to the pixel positions. The features are positioned
        # on a grid stretching from one block coord to another whereas
        # the pixel grid ends just short of the next block coordinate
        #
        # Example patch (x = pixel position, B = block coordinate position)
        # and relative coordinate positions.
        #
        # -1 ^ B x x x x x x x B
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #    | x x x x x x x x x
        #  1 v B x x x x x x x B
        #      <--------------->
        #     -1               1
        #
        # When we generate samples for a patch, we sample an
        # 8x8 grid that extends between block coords, i.e.
        # between the arrows above
        #
        if eval:
            if True or oversample != 1.:
                post = torch.linspace(-1, 1, int(self.voxel_size * oversample + 1))[:-1]
                post += (post[1] - post[0])/2

                eval_coords = torch.meshgrid(post, post, post)
                row_coords = eval_coords[0].flatten()
                col_coords = eval_coords[1].flatten()
                dep_coords = eval_coords[2].flatten()
            else:
                row_coords = self.eval_row_coords.flatten()
                col_coords = self.eval_col_coords.flatten()
                dep_coords = self.eval_dep_coords.flatten()

            rel_samples = torch.stack((row_coords, col_coords, dep_coords), dim=-1)
            abs_samples = self.block_coord[None, :] + (self.block_size[None, :]) * (rel_samples+1)/2

            return rel_samples, abs_samples, None

        else:
            if (self.old_block_coord is None or self.old_block_size is None or
               (self.old_block_coord != self.block_coord).any()
               or (self.old_block_size != self.block_size).any()):

                self.old_block_coord = self.block_coord
                self.old_block_size = self.block_size
                self.update_surface_coords()
            return self.select_near_mesh(self.num_samples**3)

    def update_surface_coords(self):
        center = (self.block_coord + 0.5 * self.block_size).cpu().numpy()
        side_length = float(self.block_size[0])

        search_radius = (side_length/2)*(3**0.5)
        indices = np.array(self.surface_tree.query_ball_point(center, search_radius))

        if indices.shape[0] == 0:
            self.near_mesh_abs_samples = np.zeros((0, 3))
            self.near_mesh_rel_samples = np.zeros((0, 3))
        else:
            coordinates = self.surface_tree.data[indices]
            in_cube = np.linalg.norm(coordinates - center, ord=np.inf, axis=1) < (side_length/2)
            self.near_mesh_abs_samples = torch.FloatTensor(coordinates[in_cube])

            self.near_mesh_rel_samples = 2 * (self.near_mesh_abs_samples - self.block_coord[None, :])/self.block_size[None, :] - 1

    def select_near_mesh(self, num_samples, jitter=True):
        if np.random.rand() > 0.9 and self.near_mesh_abs_samples.shape[0] > 0:  # Near surface
            selection_indices = torch.randint(self.near_mesh_abs_samples.shape[0], (num_samples,))
            rel_samples_mesh = self.near_mesh_rel_samples[selection_indices]
            abs_samples_mesh = self.near_mesh_abs_samples[selection_indices]

            rel_samples_mesh += torch.randn_like(rel_samples_mesh) * 0.05 * self.block_size
            abs_samples_mesh = self.block_coord[None, :] + (self.block_size[None, :]) * (rel_samples_mesh+1)/2

            return rel_samples_mesh, abs_samples_mesh, None
        else:  # Uniform
            row_coords = self.row_coords + torch.rand_like(self.row_coords) * 2./self.num_samples
            col_coords = self.col_coords + torch.rand_like(self.col_coords) * 2./self.num_samples
            dep_coords = self.dep_coords + torch.rand_like(self.dep_coords) * 2./self.num_samples

            rel_samples = torch.stack((row_coords, col_coords, dep_coords), dim=-1)
            abs_samples = self.block_coord[None, :] + (self.block_size[None, :]) * (rel_samples+1)/2

            return rel_samples, abs_samples, None
