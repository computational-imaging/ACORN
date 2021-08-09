'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
          loss_fn, pruning_fn, summary_fn, double_precision=False, clip_grad=False,
          loss_schedules=None, resume_checkpoint={}, objs_to_save={}, epochs_til_pruning=4):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # load optimizer if supplied
    if 'optimizer_state_dict' in resume_checkpoint:
        optim.load_state_dict(resume_checkpoint['optimizer_state_dict'])

    for g in optim.param_groups:
        g['lr'] = lr

    if os.path.exists(os.path.join(model_dir, 'summaries')):
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    total_steps = 0
    if 'total_steps' in resume_checkpoint:
        total_steps = resume_checkpoint['total_steps']

    start_epoch = 0
    if 'epoch' in resume_checkpoint:
        start_epoch = resume_checkpoint['epoch']

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        pbar.update(total_steps)
        train_losses = []
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_%06d.pth' % total_steps))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%06d.txt' % total_steps),
                           np.array(train_losses))
                save_dict = {'epoch': epoch,
                             'total_steps': total_steps,
                             'optimizer_state_dict': optim.state_dict()}
                save_dict.update(objs_to_save)
                torch.save(save_dict, os.path.join(checkpoints_dir, 'optim_%06d.pth' % total_steps))

            # prune
            if not epoch % epochs_til_pruning and epoch:
                pruning_fn(model, train_dataloader.dataset)

            if not (epoch + 1) % epochs_til_pruning:
                retile = False
            else:
                retile = True

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                tmp = {}
                for key, value in model_input.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value.cuda()})
                    else:
                        tmp.update({key: value})
                model_input = tmp

                tmp = {}
                for key, value in gt.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value.cuda()})
                    else:
                        tmp.update({key: value})
                gt = tmp

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                model_output = model(model_input)
                losses = loss_fn(model_output, gt, total_steps, retile=retile)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                optim.zero_grad()
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                total_steps += 1

            # after epoch
            tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

        # save model at end of epoch
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final_%06d.pth' % total_steps))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final_%06d.txt' % total_steps),
                   np.array(train_losses))
        save_dict = {'epoch': epoch,
                     'total_steps': total_steps,
                     'optimizer_state_dict': optim.state_dict()}
        save_dict.update(objs_to_save)
        torch.save(save_dict, os.path.join(checkpoints_dir, 'optim_final_%06d.pth' % total_steps))
