# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio
import utils
import training
import loss_functions
import pruning_functions
import modules
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import numpy as np
import skimage
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import torch
from time import time


p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=1e-3')
p.add_argument('--num_iters', type=int, default=100300,
               help='Number of iterations to train for.')
p.add_argument('--num_workers', type=int, default=0,
               help='number of dataloader workers.')
p.add_argument('--skip_logging', action='store_true', default=False,
               help="don't use summary function, only save loss and models")
p.add_argument('--eval', action='store_true', default=False,
               help='run evaluation')
p.add_argument('--resume', nargs=2, type=str, default=None,
               help='resume training, specify path to directory where model is stored and the iteration of ckpt.')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

# logging options
p.add_argument('--experiment_name', type=str, required=True,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--epochs_til_ckpt', type=int, default=2,
               help='Epochs until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Number of iterations until tensorboard summary is saved.')

# dataset options
p.add_argument('--res', nargs='+', type=int, default=[512],
               help='image resolution.')
p.add_argument('--dataset', type=str, default='camera', choices=['camera', 'pluto', 'tokyo', 'mars'],
               help='which dataset to use')
p.add_argument('--grayscale', action='store_true', default=False,
               help='whether to use grayscale')

# model options
p.add_argument('--patch_size', nargs='+', type=int, default=[32],
               help='patch size.')
p.add_argument('--hidden_features', type=int, default=4,
               help='hidden features in network')
p.add_argument('--hidden_layers', type=int, default=512,
               help='hidden layers in network')
p.add_argument('--w0', type=int, default=5,
               help='w0 for the siren model.')
p.add_argument('--steps_til_tiling', type=int, default=500,
               help='How often to recompute the tiling, also defines number of steps per epoch.')
p.add_argument('--max_patches', type=int, default=1024,
               help='maximum number of patches in the optimization')
p.add_argument('--model_type', type=str, default='multiscale', required=False, choices=['multiscale', 'siren', 'pe'],
               help='Type of model to evaluate, default is multiscale.')
p.add_argument('--scale_init', type=int, default=3,
               help='which scale to initialize active patches in the quadtree')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
opt = p.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

for k, v in opt.__dict__.items():
    print(k, v)


def main():
    if opt.dataset == 'camera':
        img_dataset = dataio.Camera()
    elif opt.dataset == 'pluto':
        pluto_url = "https://upload.wikimedia.org/wikipedia/commons/e/ef/Pluto_in_True_Color_-_High-Res.jpg"
        img_dataset = dataio.ImageFile('../data/pluto.jpg', url=pluto_url, grayscale=opt.grayscale)
    elif opt.dataset == 'tokyo':
        img_dataset = dataio.ImageFile('../data/tokyo.tif', grayscale=opt.grayscale)
    elif opt.dataset == 'mars':
        img_dataset = dataio.ImageFile('../data/mars.tif', grayscale=opt.grayscale)

    if len(opt.patch_size) == 1:
        opt.patch_size = 3*opt.patch_size

    # set up dataset
    coord_dataset = dataio.Patch2DWrapperMultiscaleAdaptive(img_dataset,
                                                            sidelength=opt.res,
                                                            patch_size=opt.patch_size[1:], jitter=True,
                                                            num_workers=opt.num_workers, length=opt.steps_til_tiling,
                                                            scale_init=opt.scale_init, max_patches=opt.max_patches)

    opt.num_epochs = opt.num_iters // coord_dataset.__len__()

    image_resolution = (opt.res, opt.res)

    dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1, pin_memory=True,
                            num_workers=opt.num_workers)

    if opt.resume is not None:
        path, iter = opt.resume
        iter = int(iter)
        assert(os.path.isdir(path))
        assert opt.config is not None, 'Specify config file'

    # Define the model.
    if opt.grayscale:
        out_features = 1
    else:
        out_features = 3

    if opt.model_type == 'multiscale':
        model = modules.ImplicitAdaptivePatchNet(in_features=3, out_features=out_features,
                                                 num_hidden_layers=opt.hidden_layers,
                                                 hidden_features=opt.hidden_features,
                                                 feature_grid_size=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
                                                 sidelength=opt.res,
                                                 num_encoding_functions=10,
                                                 patch_size=opt.patch_size[1:])

    elif opt.model_type == 'siren':
        model = modules.ImplicitNet(opt.res, in_features=2,
                                    out_features=out_features,
                                    num_hidden_layers=4,
                                    hidden_features=1536,
                                    mode='siren', w0=opt.w0)
    elif opt.model_type == 'pe':
        model = modules.ImplicitNet(opt.res, in_features=2,
                                    out_features=out_features,
                                    num_hidden_layers=4,
                                    hidden_features=1536,
                                    mode='pe')
    else:
        raise NotImplementedError('Only model types multiscale, siren, and pe are implemented')

    model.cuda()

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')

    # Define the loss
    loss_fn = partial(loss_functions.image_mse,
                      tiling_every=opt.steps_til_tiling,
                      dataset=coord_dataset,
                      model_type=opt.model_type)
    summary_fn = partial(utils.write_image_patch_multiscale_summary, image_resolution, opt.patch_size[1:], coord_dataset, model_type=opt.model_type, skip=opt.skip_logging)

    # Define the pruning function
    pruning_fn = partial(pruning_functions.no_pruning,
                         pruning_every=1)

    # if we are resuming from a saved checkpoint
    if opt.resume is not None:
        print('Loading checkpoints')
        model_dict = torch.load(path + '/checkpoints/' + f'model_{iter:06d}.pth')
        model.load_state_dict(model_dict)

        # load optimizers
        try:
            resume_checkpoint = {}
            optim_dict = torch.load(path + '/checkpoints/' + f'optim_{iter:06d}.pth')
            for g in optim_dict['optimizer_state_dict']['param_groups']:
                g['lr'] = opt.lr
            resume_checkpoint['optimizer_state_dict'] = optim_dict['optimizer_state_dict']
            resume_checkpoint['total_steps'] = optim_dict['total_steps']
            resume_checkpoint['epoch'] = optim_dict['epoch']

            # initialize model state_dict
            print('Initializing models')
            coord_dataset.quadtree.__load__(optim_dict['quadtree'])
            coord_dataset.synchronize()

        except FileNotFoundError:
            print('Unable to load optimizer checkpoints')
    else:
        resume_checkpoint = {}

    if opt.eval:
        run_eval(model, coord_dataset)
    else:
        # Save command-line parameters log directory.
        root_path = os.path.join(opt.logging_root, opt.experiment_name)
        utils.cond_mkdir(root_path)
        p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])

        # Save text summary of model into log directory.
        with open(os.path.join(root_path, "model.txt"), "w") as out_file:
            out_file.write(str(model))

        objs_to_save = {'quadtree': coord_dataset.quadtree}

        training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                       steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                       model_dir=root_path, loss_fn=loss_fn, pruning_fn=pruning_fn, summary_fn=summary_fn, objs_to_save=objs_to_save,
                       resume_checkpoint=resume_checkpoint)


# evaluate PSNR at saved checkpoints and save model outputs
def run_eval(model, coord_dataset):
    # get checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(opt.config), 'checkpoints')

    # make eval directory
    eval_dir = os.path.join(os.path.dirname(opt.config), 'eval')
    utils.cond_mkdir(eval_dir)

    # get model & optim files
    model_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'model_[0-9]+.pth', f)], reverse=True)
    optim_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'optim_[0-9]+.pth', f)], reverse=True)

    # extract iterations
    iters = [int(re.search(r'[0-9]+', f)[0]) for f in model_files]

    # append beginning of path
    model_files = [os.path.join(checkpoint_dir, f) for f in model_files]
    optim_files = [os.path.join(checkpoint_dir, f) for f in optim_files]

    # iterate through model and optim files
    metrics = {}
    saved_gt = False
    for curr_iter, model_path, optim_path in zip(tqdm(iters), model_files, optim_files):

        # load model and optimizer files
        print('Loading models')
        model_dict = torch.load(model_path)
        optim_dict = torch.load(optim_path)

        # initialize model state_dict
        print('Initializing models')
        model.load_state_dict(model_dict)
        coord_dataset.quadtree.__load__(optim_dict['quadtree'])
        coord_dataset.synchronize()

        # save image and calculate psnr
        coord_dataset.toggle_eval()
        model_input, gt = coord_dataset[0]
        coord_dataset.toggle_eval()

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
        print('Running forward pass')
        n_channels = gt['img'].shape[-1]
        start = time()
        with torch.no_grad():
            pred_img = utils.process_batch_in_chunks(model_input, model, max_chunk_size=512)['model_out']['output']
        torch.cuda.synchronize()
        print(f'Model: {time() - start:.02f}')

        # get pixel idx for each coordinate
        start = time()
        coords = model_input['fine_abs_coords'].detach().cpu().numpy()
        pixel_idx = np.zeros_like(coords).astype(np.int32)
        pixel_idx[..., 0] = np.round((coords[..., 0] + 1.)/2. * (coord_dataset.sidelength[0]-1)).astype(np.int32)
        pixel_idx[..., 1] = np.round((coords[..., 1] + 1.)/2. * (coord_dataset.sidelength[1]-1)).astype(np.int32)
        pixel_idx = pixel_idx.reshape(-1, 2)

        # assign predicted image values into a new array
        # need to use numpy since it supports index assignment
        pred_img = pred_img.detach().cpu().numpy().reshape(-1, n_channels)
        display_pred = np.zeros((*coord_dataset.sidelength, n_channels))
        display_pred[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = pred_img
        display_pred = torch.tensor(display_pred)[None, ...]
        display_pred = display_pred.permute(0, 3, 1, 2)

        if not saved_gt:
            gt_img = gt['img'].detach().cpu().numpy().reshape(-1, n_channels)
            display_gt = np.zeros((*coord_dataset.sidelength, n_channels))
            display_gt[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = gt_img
            display_gt = torch.tensor(display_gt)[None, ...]
            display_gt = display_gt.permute(0, 3, 1, 2)
        print(f'Reshape: {time() - start:.02f}')

        # record metrics
        start = time()
        psnr, ssim = get_metrics(display_pred, display_gt)
        metrics.update({curr_iter: {'psnr': psnr, 'ssim': ssim}})
        print(f'Metrics: {time() - start:.02f}')
        print(f'Iter: {curr_iter}, PSNR: {psnr:.02f}')

        # save images
        pred_out = np.clip((display_pred.squeeze().numpy()/2.) + 0.5, a_min=0., a_max=1.).transpose(1, 2, 0)*255
        pred_out = pred_out.astype(np.uint8)
        pred_fname = os.path.join(eval_dir, f'pred_{curr_iter:06d}.png')
        print('Saving image')
        cv2.imwrite(pred_fname, cv2.cvtColor(pred_out, cv2.COLOR_RGB2BGR))

        if not saved_gt:
            print('Saving gt')
            gt_out = np.clip((display_gt.squeeze().numpy()/2.) + 0.5, a_min=0., a_max=1.).transpose(1, 2, 0)*255
            gt_out = gt_out.astype(np.uint8)
            gt_fname = os.path.join(eval_dir, 'gt.png')
            cv2.imwrite(gt_fname, cv2.cvtColor(gt_out, cv2.COLOR_RGB2BGR))
            saved_gt = True

        # save tiling
        tiling_fname = os.path.join(eval_dir, f'tiling_{curr_iter:06d}.pdf')
        coord_dataset.quadtree.draw()
        plt.savefig(tiling_fname)

        # save metrics
        metric_fname = os.path.join(eval_dir, f'metrics_{curr_iter:06d}.npy')
        np.save(metric_fname, metrics)


def get_metrics(pred_img, gt_img):
    pred_img = pred_img.detach().cpu().numpy().squeeze()
    gt_img = gt_img.detach().cpu().numpy().squeeze()

    p = pred_img.transpose(1, 2, 0)
    trgt = gt_img.transpose(1, 2, 0)

    p = (p / 2.) + 0.5
    p = np.clip(p, a_min=0., a_max=1.)

    trgt = (trgt / 2.) + 0.5

    ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

    return psnr, ssim


if __name__ == '__main__':
    main()
