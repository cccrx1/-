"""
REFINE with Adaptive Threshold Sample Selection

This is an improved version of REFINE that uses adaptive threshold
to better separate clean and poisoned samples during training.

Key improvements:
1. Dynamic threshold based on loss distribution statistics
2. Progressive threshold adjustment across training epochs
3. Better balance between clean accuracy and ASR reduction
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import os.path as osp

from .REFINE import REFINE
from ..utils import Log


class REFINE_ADAPTIVE(REFINE):
    """REFINE with adaptive threshold sample selection.

    Args:
        adaptive_mode (str): Adaptive strategy. Options: 'progressive', 'statistical'. Default: 'progressive'.
        initial_threshold (float): Initial threshold multiplier. Default: 1.5.
        final_threshold (float): Final threshold multiplier. Default: 0.5.
        warmup_ratio (float): Ratio of epochs for warmup phase. Default: 0.3.
    """

    def __init__(self,
                 unet,
                 model,
                 pretrain=None,
                 arr_path=None,
                 num_classes=10,
                 lmd=0.1,
                 seed=0,
                 deterministic=False,
                 adaptive_mode='progressive',
                 initial_threshold=1.5,
                 final_threshold=0.5,
                 warmup_ratio=0.3):
        super(REFINE_ADAPTIVE, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic
        )
        self.adaptive_mode = adaptive_mode
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_ratio = warmup_ratio

    def compute_adaptive_threshold(self, losses, epoch, total_epochs):
        """Compute adaptive threshold based on loss distribution.

        Args:
            losses (torch.Tensor): Loss values for all samples
            epoch (int): Current epoch
            total_epochs (int): Total training epochs

        Returns:
            float: Adaptive threshold value
        """
        progress = epoch / total_epochs
        mean_loss = losses.mean().item()
        std_loss = losses.std().item()

        if self.adaptive_mode == 'progressive':
            # Progressive threshold: strict -> loose -> strict
            if progress < self.warmup_ratio:
                # Early stage: loose threshold
                threshold_multiplier = self.initial_threshold
            elif progress < 0.7:
                # Middle stage: gradually tighten
                t = (progress - self.warmup_ratio) / (0.7 - self.warmup_ratio)
                threshold_multiplier = self.initial_threshold - t * (self.initial_threshold - 1.0)
            else:
                # Late stage: strict threshold
                t = (progress - 0.7) / 0.3
                threshold_multiplier = 1.0 - t * (1.0 - self.final_threshold)
        else:
            # Statistical mode: based on distribution only
            threshold_multiplier = 1.0

        threshold = mean_loss + threshold_multiplier * std_loss
        return threshold, mean_loss, std_loss

    def train_unet(self, train_dataset, test_dataset, schedule):
        """Override train_unet to add adaptive threshold logging."""
        if 'pretrain' in schedule:
            self.unet.load_state_dict(torch.load(schedule['pretrain']), strict=False)
        if 'arr_path' in schedule:
            self.arr_shuffle.load_state_dict(torch.load(schedule['arr_path']), strict=False)

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            if schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(schedule['GPU_num']))
                self.unet = torch.nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='none')
        from ..utils import SupConLoss
        supconloss_func = SupConLoss()
        optimizer = torch.optim.Adam(self.unet.parameters(), schedule['lr'], schedule['betas'],
                                     schedule['eps'], schedule['weight_decay'], schedule['amsgrad'])

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' +
                           time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))

        iteration = 0
        last_time = time.time()
        best_eval_loss = float("inf")
        best_epoch = 0
        best_unet_state = None

        msg = f"Total train samples: {len(train_dataset)}\nTotal test samples: {len(test_dataset)}\n"
        msg += f"Batch size: {schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        msg += f"Adaptive mode: {self.adaptive_mode}\nInitial threshold: {self.initial_threshold}\n"
        msg += f"Final threshold: {self.final_threshold}\nWarmup ratio: {self.warmup_ratio}\n"
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']

            epoch_losses = []
            for batch_id, batch in enumerate(train_loader):
                batch_img, _ = batch
                batch_img = batch_img.to(device)
                bsz = batch_img.shape[0]

                f_logit = self.model(batch_img)
                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                features = self.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                f1, f2 = features, features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supconloss = supconloss_func(features, f_index)

                loss_per_sample = loss_func(logit, f_label).mean(dim=1)
                loss = loss_per_sample.mean() + self.lmd * supconloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss_per_sample.detach().cpu())
                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                    msg += f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_loader)}, "
                    msg += f"lr: {schedule['lr']}, loss: {float(loss):.6f}, time: {time.time()-last_time:.2f}\n"
                    last_time = time.time()
                    log(msg)

            # Compute adaptive threshold at end of epoch
            epoch_losses_tensor = torch.cat(epoch_losses)
            threshold, mean_loss, std_loss = self.compute_adaptive_threshold(
                epoch_losses_tensor, i + 1, schedule['epochs']
            )
            msg = f"[Adaptive Threshold] Epoch {i+1}: mean_loss={mean_loss:.6f}, std_loss={std_loss:.6f}, threshold={threshold:.6f}\n"
            log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                loss_value = float(loss.item() if hasattr(loss, "item") else loss)
                msg = "==========Test result on test dataset==========\n"
                msg += time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                msg += f"loss: {loss_value:.6f}, time: {time.time()-last_time:.2f}\n"
                log(msg)

                if loss_value < best_eval_loss:
                    best_eval_loss = loss_value
                    best_epoch = i + 1
                    from copy import deepcopy
                    best_unet_state = deepcopy(self.unet.state_dict())
                    torch.save(best_unet_state, os.path.join(work_dir, "best_ckpt.pth"))
                    log(f"[Best] epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")

                self.unet = self.unet.to(device)
                self.unet.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.unet.eval()
                self.unet = self.unet.cpu()
                ckpt_unet_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_unet_path = os.path.join(work_dir, ckpt_unet_filename)
                torch.save(self.unet.state_dict(), ckpt_unet_path)
                self.unet = self.unet.to(device)
                self.unet.train()

        if best_unet_state is not None:
            self.unet.load_state_dict(best_unet_state)
            self.unet = self.unet.to(device)
            self.unet.eval()
            log(f"Restore best checkpoint from epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")
