import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, momentum_schedule=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():

            with torch.no_grad():
                teacher_latent, teacher_ids_keep, student_ids_keep = teacher(samples, mask_ratio=args.mask_ratio, 
                        drop_stage=args.drop_stage, random_drop=args.random_drop, drop_case=args.drop_case, 
                        keep_ratio=args.keep_ratio)
                
                teacher_latent = teacher_latent.detach()

                m = teacher_latent.mean(-1, keepdims=True)
                s = teacher_latent.var(-1, keepdims=True)
                teacher_latent = (teacher_latent - m) / (s + 1e-6) ** 0.5
            
            latent = model(samples, student_ids_keep=student_ids_keep, teacher_ids_keep=teacher_ids_keep)

            loss = 1 - F.cosine_similarity(latent, teacher_latent, dim=-1).mean()
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
        m = momentum_schedule[len(data_loader) * epoch + data_iter_step]
        with torch.no_grad():
            student_param = {k: v for k, v in model.named_parameters()}
            for k, v in teacher.named_parameters():
                v.data.mul_(m).add_((1 - m) * student_param[k].detach().data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
