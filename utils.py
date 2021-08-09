import numpy as np
import torch
import os
from torchvision.utils import make_grid
import skimage.measure
from tqdm import tqdm
import mrcfile


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs = list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)
        trgt = (trgt / 2.) + 0.5

        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
        psnrs.append(psnr)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)


def write_occupancy_multiscale_summary(image_resolution, dataset, model, model_input, gt,
                                       model_output, writer, total_steps, prefix='train_',
                                       output_mrc='test.mrc', skip=False,
                                       oversample=1.0, max_chunk_size=1024, mode='binary'):
    if skip:
        return

    model_input = dataset.get_eval_samples(oversample)

    print("Summary: Write occupancy multiscale summary...")

    # convert to cuda and add batch dimension
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...]})
        else:
            tmp.update({key: value})
    model_input = tmp

    print("Summary: processing...")
    pred_occupancy = process_batch_in_chunks(model_input, model, max_chunk_size=max_chunk_size)['model_out']['output']

    # get voxel idx for each coordinate
    coords = model_input['fine_abs_coords'].detach().cpu().numpy()
    voxel_idx = np.floor((coords + 1.) / 2. * (dataset.sidelength[0] * oversample)).astype(np.int32)
    voxel_idx = voxel_idx.reshape(-1, 3)

    # init a new occupancy volume
    display_occupancy = -1 * np.ones(image_resolution, dtype=np.float32)

    # assign predicted voxel occupancy values into the array
    pred_occupancy = pred_occupancy.reshape(-1, 1).detach().cpu().numpy()
    display_occupancy[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = pred_occupancy[..., 0]

    print(f"Summary: write MRC file {image_resolution}")
    if mode == 'hq':
        print("\tWriting float")
        with mrcfile.new_mmap(output_mrc, overwrite=True, shape=image_resolution, mrc_mode=2) as mrc:
            mrc.data[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = pred_occupancy[..., 0]
    elif mode == 'binary':
        print("\tWriting binary")
        with mrcfile.new_mmap(output_mrc, overwrite=True, shape=image_resolution) as mrc:
            mrc.data[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = pred_occupancy[..., 0] > 0

    if writer is not None:
        print("Summary: Draw octtree")
        fig = dataset.octtree.draw()
        writer.add_figure(prefix + 'tiling', fig, global_step=total_steps)

    return display_occupancy


def write_image_patch_multiscale_summary(image_resolution, patch_size, dataset, model, model_input, gt,
                                         model_output, writer, total_steps, prefix='train_',
                                         model_type='multiscale', skip=False):
    if skip:
        return

    # uniformly sample the image
    dataset.toggle_eval()
    model_input, gt = dataset[0]
    dataset.toggle_eval()

    # convert to cuda and add batch dimension
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cpu()})
        else:
            tmp.update({key: value})
    model_input = tmp

    tmp = {}
    for key, value in gt.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cpu()})
        else:
            tmp.update({key: value})
    gt = tmp

    # run the model on uniform samples
    n_channels = gt['img'].shape[-1]
    pred_img = process_batch_in_chunks(model_input, model)['model_out']['output']

    # get pixel idx for each coordinate
    coords = model_input['fine_abs_coords'].detach().cpu().numpy()
    pixel_idx = np.zeros_like(coords).astype(np.int32)
    pixel_idx[..., 0] = np.round((coords[..., 0] + 1.)/2. * (dataset.sidelength[0]-1)).astype(np.int32)
    pixel_idx[..., 1] = np.round((coords[..., 1] + 1.)/2. * (dataset.sidelength[1]-1)).astype(np.int32)
    pixel_idx = pixel_idx.reshape(-1, 2)

    # get pixel idx for each coordinate in frozen patches
    frozen_coords, frozen_values = dataset.get_frozen_patches()
    if frozen_coords is not None:
        frozen_coords = frozen_coords.detach().cpu().numpy()
        frozen_pixel_idx = np.zeros_like(frozen_coords).astype(np.int32)
        frozen_pixel_idx[..., 0] = np.round((frozen_coords[..., 0] + 1.) / 2. * (dataset.sidelength[0] - 1)).astype(np.int32)
        frozen_pixel_idx[..., 1] = np.round((frozen_coords[..., 1] + 1.) / 2. * (dataset.sidelength[1] - 1)).astype(np.int32)
        frozen_pixel_idx = frozen_pixel_idx.reshape(-1, 2)

    # init a new reconstructed image
    display_pred = np.zeros((*dataset.sidelength, n_channels))

    # assign predicted image values into a new array
    # need to use numpy since it supports index assignment
    pred_img = pred_img.reshape(-1, n_channels).detach().cpu().numpy()
    display_pred[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = pred_img

    # assign frozen image values into the array too
    if frozen_coords is not None:
        frozen_values = frozen_values.reshape(-1, n_channels).detach().cpu().numpy()
        display_pred[[frozen_pixel_idx[:, 0]], [frozen_pixel_idx[:, 1]]] = frozen_values

    # show reconstructed img
    display_pred = torch.tensor(display_pred)[None, ...]
    display_pred = display_pred.permute(0, 3, 1, 2)

    gt_img = gt['img'].reshape(-1, n_channels).detach().cpu().numpy()
    display_gt = np.zeros((*dataset.sidelength, n_channels))
    display_gt[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = gt_img
    display_gt = torch.tensor(display_gt)[None, ...]
    display_gt = display_gt.permute(0, 3, 1, 2)

    fig = dataset.quadtree.draw()
    writer.add_figure(prefix + 'tiling', fig, global_step=total_steps)

    if 'img' in gt:
        output_vs_gt = torch.cat((display_gt, display_pred), dim=0)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)
        write_psnr(display_pred, display_gt, writer, total_steps, prefix+'img_')


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp


def process_batch_in_chunks(in_dict, model, max_chunk_size=1024, progress=None):

    in_chunked = []
    for key in in_dict:
        chunks = torch.split(in_dict[key], max_chunk_size, dim=1)
        in_chunked.append(chunks)

    list_chunked_batched_in = \
        [{k: v for k, v in zip(in_dict.keys(), curr_chunks)} for curr_chunks in zip(*in_chunked)]
    del in_chunked

    list_chunked_batched_out_out = {}
    list_chunked_batched_out_in = {}
    for chunk_batched_in in tqdm(list_chunked_batched_in):
        chunk_batched_in = {k: v.cuda() for k, v in chunk_batched_in.items()}
        tmp = model(chunk_batched_in)
        tmp = dict2cpu(tmp)

        for key in tmp['model_out']:
            if tmp['model_out'][key] is None:
                continue

            out_ = tmp['model_out'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_out.setdefault(key, []).append(out_)

        for key in tmp['model_in']:
            if tmp['model_in'][key] is None:
                continue

            in_ = tmp['model_in'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_in.setdefault(key, []).append(in_)

        del tmp, chunk_batched_in

    # Reassemble the output chunks in a batch
    batched_out = {}
    for key in list_chunked_batched_out_out:
        batched_out_lin = torch.cat(list_chunked_batched_out_out[key], dim=1)
        batched_out[key] = batched_out_lin

    batched_in = {}
    for key in list_chunked_batched_out_in:
        batched_in_lin = torch.cat(list_chunked_batched_out_in[key], dim=1)
        batched_in[key] = batched_in_lin

    return {'model_in': batched_in, 'model_out': batched_out}


def subsample_dict(in_dict, num_views, multiscale=False):
    if multiscale:
        out = {}
        for k, v in in_dict.items():
            if v.shape[0] == in_dict['octant_coords'].shape[0]:
                # this is arranged by blocks
                out.update({k: v[0:num_views[0]]})
            else:
                # arranged by rays
                out.update({k: v[0:num_views[1]]})
    else:
        out = {key: value[0:num_views, ...] for key, value in in_dict.items()}

    return out
