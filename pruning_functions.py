import torch
import utils


def no_pruning(model, dataset, pruning_every=100):
    return


def pruning_occupancy(model, dataset, threshold=-10):
    model_input = dataset.get_eval_samples(1)

    print("Pruning: loading data to cuda...")
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cuda()})
        else:
            tmp.update({key: value})
    model_input = tmp

    print("Pruning: evaluating occupancy...")
    pred_occupancy = utils.process_batch_in_chunks(model_input, model)['model_out']['output']
    pred_occupancy = torch.max(pred_occupancy, dim=-2).values.squeeze()
    pred_occupancy_idx = model_input['coord_octant_idx'].squeeze()

    print("Pruning: computing mean and freezing empty octants")
    active_octants = dataset.octtree.get_active_octants()

    frozen_octants = 0
    for idx, octant in enumerate(active_octants):
        max_prediction = torch.max(pred_occupancy[pred_occupancy_idx == idx])
        if max_prediction < threshold and octant.err < 1e-3:  # Prune if model is confident that everything is empty
            octant.frozen = True
            frozen_octants += 1
    print(f"Pruning: Froze {frozen_octants} octants.")
    dataset.synchronize()
