import torch


def image_mse(model_output, gt, step, tiling_every=100, dataset=None, model_type='multiscale', retile=True):
    img_loss = (model_output['model_out']['output'] - gt['img'])**2

    if model_type == 'multiscale':
        per_patch_loss = torch.mean(img_loss, dim=(-1, -2)).squeeze(0).detach().cpu().numpy()

        dataset.update_patch_err(per_patch_loss, step)
        if step % tiling_every == tiling_every-1 and retile:
            tiling_stats = dataset.update_tiling()
            if tiling_stats['merged'] != 0 or tiling_stats['splits'] != 0:
                dataset.synchronize()

    return {'img_loss': img_loss.mean()}


def occupancy_bce(model_output, gt, step, tiling_every=100, dataset=None,
                  model_type='multiscale', pruning_fn=None, retile=True):
    occupancy_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(model_output['model_out']['output'], gt['occupancy'].float())

    if model_type == 'multiscale':
        per_octant_loss = torch.mean(occupancy_loss, dim=(-1, -2)).squeeze(0).detach().cpu().numpy()

        dataset.update_octant_err(per_octant_loss, step)
        if step % tiling_every == tiling_every-1 and retile:
            tiling_stats = dataset.update_tiling()
            if tiling_stats['merged'] != 0 or tiling_stats['splits'] != 0:
                dataset.synchronize()

    return {'occupancy_loss': occupancy_loss.mean()}
