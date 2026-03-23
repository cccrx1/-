'''
REFINE-SSL: self-supervised contrastive variant of REFINE.
This variant removes pseudo-label supervised contrastive loss and
uses SimCLR-style self-supervised contrastive loss.
'''

import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from .REFINE import REFINE
from ..utils import Log


class REFINE_SSL(REFINE):
    """REFINE with self-supervised contrastive training objective."""

    def __init__(self,
                 unet,
                 model,
                 pretrain=None,
                 arr_path=None,
                 num_classes=10,
                 lmd=0.1,
                 seed=0,
                 deterministic=False,
                 temperature=0.07,
                 selfsup_weight=0.02):
        super(REFINE_SSL, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic,
        )
        self.temperature = float(temperature)
        self.selfsup_weight = float(selfsup_weight)

    @staticmethod
    def _simclr_aug(sample):
        sample = TF.pad(sample, padding=[4, 4, 4, 4], fill=0)
        i, j, h, w = torch.randint(0, 9, (2,), device=sample.device).tolist() + [32, 32]
        sample = TF.crop(sample, i, j, h, w)
        if torch.rand(1, device=sample.device).item() < 0.5:
            sample = TF.hflip(sample)
        return sample

    def _selfsup_contrastive_loss(self, batch_img):
        x1 = torch.stack([self._simclr_aug(sample) for sample in batch_img], dim=0)
        x2 = torch.stack([self._simclr_aug(sample) for sample in batch_img], dim=0)

        z1 = torch.clamp(self.unet(x1), 0, 1).view(batch_img.shape[0], -1)
        z2 = torch.clamp(self.unet(x2), 0, 1).view(batch_img.shape[0], -1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        features = torch.cat([z1, z2], dim=0)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

        n = batch_img.shape[0]
        labels = torch.arange(2 * n, device=batch_img.device)
        labels = (labels + n) % (2 * n)

        diag_mask = torch.eye(2 * n, device=batch_img.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, -1e9)

        return F.cross_entropy(logits, labels)

    def _test(self, dataset, device, batch_size=16, num_workers=8, loss_func=torch.nn.BCELoss(reduction='none')):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)

            losses = []
            for batch in test_loader:
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                f_logit = self.model(batch_img)
                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                ce_loss = loss_func(logit, f_label).mean()
                loss_self = self._selfsup_contrastive_loss(batch_img)
                total_loss = ce_loss + self.selfsup_weight * loss_self

                losses.append(total_loss.detach().cpu().view(1))

            losses = torch.cat(losses, dim=0)
            return losses.mean()

    def train_unet(self, train_dataset, test_dataset, schedule):
        if 'pretrain' in schedule:
            self.unet.load_state_dict(torch.load(schedule['pretrain']), strict=False)
        if 'arr_path' in schedule:
            self.arr_shuffle.load_state_dict(torch.load(schedule['arr_path']), strict=False)

        if 'device' in schedule and schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

            if schedule['GPU_num'] == 1:
                device = torch.device('cuda:0')
            else:
                gpus = list(range(schedule['GPU_num']))
                self.unet = torch.nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
        else:
            device = torch.device('cpu')

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

        loss_func = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.unet.parameters(), schedule['lr'], schedule['betas'], schedule['eps'], schedule['weight_decay'], schedule['amsgrad'])

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(train_dataset)}\nTotal test samples: {len(test_dataset)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']

            for batch_id, batch in enumerate(train_loader):
                batch_img, _ = batch
                batch_img = batch_img.to(device)

                f_logit = self.model(batch_img)
                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).to(device).scatter_(1, f_index.view(-1, 1), 1)

                logit = self.forward(batch_img)

                cross_entropy_loss = loss_func(logit, f_label)
                loss_self = self._selfsup_contrastive_loss(batch_img)
                total_loss = cross_entropy_loss + self.selfsup_weight * loss_self

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) + \
                          f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset)//schedule['batch_size']}, lr: {schedule['lr']}, loss: {float(total_loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                msg = '==========Test result on test dataset==========\n' + \
                      time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) + \
                      f'loss: {loss}, time: {time.time()-last_time}\n'
                log(msg)

                self.unet = self.unet.to(device)
                self.unet.train()

            if (i + 1) % schedule['save_epoch_interval'] == 0:
                self.unet.eval()
                self.unet = self.unet.cpu()
                ckpt_unet_filename = 'ckpt_epoch_' + str(i + 1) + '.pth'
                ckpt_unet_path = os.path.join(work_dir, ckpt_unet_filename)
                torch.save(self.unet.state_dict(), ckpt_unet_path)
                self.unet = self.unet.to(device)
                self.unet.train()