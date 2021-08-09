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
import torch
import re
import mcubes


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')
p.add_argument('--num_workers', type=int, default=0,
               help='number of dataloader workers.')
p.add_argument('--pc_filepath', type=str, default='', help='pc_or_mesh')
p.add_argument('--load', type=str, default=None, help='logging directory to resume from')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

# logging options
p.add_argument('--experiment_name', type=str, required=True,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--skip_logging', action='store_true', default=False,
               help="don't use summary function, only save loss and models")
p.add_argument('--epochs_til_ckpt', type=int, default=4,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')

# dataset options
p.add_argument('--res', type=int, default=512,
               help='image resolution.')
p.add_argument('--octant_size', type=int, default=4,
               help='patch size.')
p.add_argument('--max_octants', type=int, default=1024)
p.add_argument('--scale_init', type=int, default=3,
               help='which scale to initialize active octants in the octree')

# model options
p.add_argument('--steps_til_tiling', type=int, default=1000,
               help='how often to recompute the tiling')
p.add_argument('--hidden_features', type=int, default=512,
               help='hidden features in network')
p.add_argument('--hidden_layers', type=int, default=4,
               help='hidden layers in network')
p.add_argument('--feature_grid_size', nargs='+', type=int, default=(18, 12, 12, 12))
p.add_argument('--pruning_threshold', type=float, default=-10)
p.add_argument('--epochs_til_pruning', type=int, default=4)

# export options
p.add_argument('--export', action='store_true', default=False,
               help='export mesh from checkpoint (requires load flag)')
p.add_argument('--upsample', type=int, default=2,
               help='how much to upsamples the occupancies used to generate the output mesh')
p.add_argument('--mc_threshold', type=float, default=0.005, help='threshold for marching cubes')


opt = p.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

feature_grid_size = tuple(opt.feature_grid_size)

assert opt.pc_filepath is not None, "Must specify dataset input"

for k, v in opt.__dict__.items():
    print(k, v)
print()


def main():
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    point_cloud_dataset = dataio.OccupancyDataset(opt.pc_filepath)

    coord_dataset = dataio.Block3DWrapperMultiscaleAdaptive(point_cloud_dataset,
                                                            sidelength=opt.res,
                                                            octant_size=opt.octant_size,
                                                            jitter=True, max_octants=opt.max_octants,
                                                            num_workers=opt.num_workers,
                                                            length=opt.steps_til_tiling,
                                                            scale_init=opt.scale_init)

    model = modules.ImplicitAdaptiveOctantNet(in_features=3+1, out_features=1,
                                              num_hidden_layers=opt.hidden_layers,
                                              hidden_features=opt.hidden_features,
                                              feature_grid_size=feature_grid_size,
                                              octant_size=opt.octant_size)
    model.cuda()

    resume_checkpoint = {}
    if opt.load is not None:
        resume_checkpoint = load_from_checkpoint(opt.load, model, coord_dataset)

    if opt.export:
        assert opt.load is not None, 'Need to specify which model to export with --load'

        export_mesh(model, coord_dataset, opt.upsample, opt.mc_threshold)
        return

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n\nTrainable Parameters: {num_params}\n\n")

    dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1, pin_memory=True,
                            num_workers=opt.num_workers)

    # Define the loss
    loss_fn = partial(loss_functions.occupancy_bce,
                      tiling_every=opt.steps_til_tiling,
                      dataset=coord_dataset)

    summary_fn = partial(utils.write_occupancy_multiscale_summary, (opt.res, opt.res, opt.res),
                         coord_dataset, output_mrc=f'{opt.experiment_name}.mrc',
                         skip=opt.skip_logging)

    # Define the pruning
    pruning_fn = partial(pruning_functions.pruning_occupancy,
                         threshold=opt.pruning_threshold)

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    objs_to_save = {'octtree': coord_dataset.octtree}

    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, objs_to_save=objs_to_save,
                   pruning_fn=pruning_fn, epochs_til_pruning=opt.epochs_til_pruning,
                   resume_checkpoint=resume_checkpoint)


def load_from_checkpoint(experiment_dir, model, coord_dataset):
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    model_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'model_[0-9]+.pth', f)], reverse=False)
    optim_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'optim_[0-9]+.pth', f)], reverse=False)

    # append beginning of path
    model_files = [os.path.join(checkpoint_dir, f) for f in model_files]
    optim_files = [os.path.join(checkpoint_dir, f) for f in optim_files]
    model_path = model_files[-1]
    optim_path = optim_files[-1]

    print("MODEL PATH: ", model_path)

    # load model and octree
    print('Loading models')
    model_dict = torch.load(model_path)
    optim_dict = torch.load(optim_path)

    # initialize model state_dict
    print('Initializing models')
    model.load_state_dict(model_dict)
    coord_dataset.octtree.__load__(optim_dict['octtree'])
    coord_dataset.synchronize()

    resume_checkpoint = {}
    resume_checkpoint['optimizer_state_dict'] = optim_dict['optimizer_state_dict']
    resume_checkpoint['total_steps'] = optim_dict['total_steps']
    resume_checkpoint['epoch'] = optim_dict['epoch']

    return resume_checkpoint


def export_mesh(model, dataset, upsample, mcubes_threshold=0.005):
    res = 3*(upsample*opt.res,)
    model.octant_size = model.octant_size * upsample

    print('Export: calculating occupancy...')
    mrc_fname = os.path.join(opt.logging_root, opt.experiment_name, f"{opt.experiment_name}.mrc")
    occupancy = utils.write_occupancy_multiscale_summary(res, dataset, model,
                                                         None, None, None, None, None,
                                                         output_mrc=mrc_fname,
                                                         oversample=upsample,
                                                         mode='hq')

    print('Export: running marching cubes...')
    vertices, faces = mcubes.marching_cubes(occupancy, mcubes_threshold)

    print('Export: exporting mesh...')
    out_fname = os.path.join(opt.logging_root, opt.experiment_name, f"{opt.experiment_name}.dae")
    mcubes.export_mesh(vertices, faces, out_fname)


if __name__ == '__main__':
    main()
