import os
import os.path as osp
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from .REFINE import REFINE
from ..utils.log import Log
from ..utils.supconloss import SupConLoss


class _PDBMixin:
    """Reusable proactive defensive-trigger primitives for REFINE variants."""

    def _init_pdb(
        self,
        pdb_trigger_type=1,
        pdb_pix_value=1.0,
        pdb_target_shift=1,
        pdb_weight=0.5,
        pdb_batch_ratio=0.5,
        pdb_apply_inference_trigger=True,
        pdb_warmup_ratio=0.3,
        ssl_warmup_ratio=0.3,
        aux_loss_cap_ratio=1.5,
    ):
        self.pdb_trigger_type = int(pdb_trigger_type)
        self.pdb_pix_value = float(pdb_pix_value)
        self.pdb_target_shift = int(pdb_target_shift)
        self.pdb_weight = float(pdb_weight)
        self.pdb_batch_ratio = float(pdb_batch_ratio)
        self.pdb_apply_inference_trigger = bool(pdb_apply_inference_trigger)
        self.pdb_warmup_ratio = max(0.0, float(pdb_warmup_ratio))
        self.ssl_warmup_ratio = max(0.0, float(ssl_warmup_ratio))
        self.aux_loss_cap_ratio = max(0.0, float(aux_loss_cap_ratio))

    @staticmethod
    def _aux_progress_scale(current_epoch: int, total_epochs: int, warmup_ratio: float) -> float:
        if warmup_ratio <= 0.0 or total_epochs <= 0:
            return 1.0
        warmup_epochs = max(1, int(round(total_epochs * warmup_ratio)))
        return min(1.0, float(current_epoch + 1) / float(warmup_epochs))

    def _effective_aux_weight(self, base_weight: float, current_epoch: int, total_epochs: int, warmup_ratio: float) -> float:
        return float(base_weight) * self._aux_progress_scale(current_epoch, total_epochs, warmup_ratio)

    def _cap_aux_loss(self, aux_loss: torch.Tensor, ce_loss: torch.Tensor) -> torch.Tensor:
        if self.aux_loss_cap_ratio <= 0.0:
            return aux_loss
        cap = ce_loss.detach() * self.aux_loss_cap_ratio
        return torch.minimum(aux_loss, cap)

    def _trigger_mask(self, images: torch.Tensor) -> torch.Tensor:
        _, c, h, w = images.shape
        mask = torch.zeros((1, c, h, w), dtype=images.dtype, device=images.device)

        if self.pdb_trigger_type == 0:
            border = 1
            mask[:, :, :border, :] = 1
            mask[:, :, -border:, :] = 1
            mask[:, :, :, :border] = 1
            mask[:, :, :, -border:] = 1
        elif self.pdb_trigger_type == 1:
            patch = min(7, h, w)
            mask[:, :, :patch, :patch] = 1
        elif self.pdb_trigger_type == 2:
            mask[:, :, ::2, 0] = 1
            mask[:, :, ::2, -1] = 1
            mask[:, :, 0, ::2] = 1
            mask[:, :, -1, ::2] = 1
        else:
            raise ValueError(f"Invalid pdb_trigger_type={self.pdb_trigger_type}")

        return mask

    def _apply_defensive_trigger(self, images: torch.Tensor) -> torch.Tensor:
        mask = self._trigger_mask(images)
        trigger_value = torch.full_like(images, self.pdb_pix_value)
        triggered = images * (1.0 - mask) + trigger_value * mask
        return torch.clamp(triggered, 0.0, 1.0)

    def _defensive_target(self, pseudo_index: torch.Tensor) -> torch.Tensor:
        return (pseudo_index + self.pdb_target_shift) % self.num_classes

    def _inverse_shift_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # h(y)=y+s, so h^{-1}(k)=k-s; gather equivalent column order once.
        gather_index = (torch.arange(self.num_classes, device=probs.device) + self.pdb_target_shift) % self.num_classes
        return probs[:, gather_index]

    def _sample_defensive_subset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.pdb_batch_ratio >= 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if self.pdb_batch_ratio <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        sampled = torch.rand(batch_size, device=device) < self.pdb_batch_ratio
        if sampled.sum() == 0:
            sampled[torch.randint(0, batch_size, (1,), device=device)] = True
        return sampled

    def _pdb_guidance_loss(self, purified: torch.Tensor, pseudo_index: torch.Tensor) -> torch.Tensor:
        selected = self._sample_defensive_subset(purified.shape[0], purified.device)
        if selected.sum() == 0:
            return torch.zeros((), device=purified.device, dtype=purified.dtype)

        triggered = self._apply_defensive_trigger(purified[selected])
        logits = self.model(triggered)
        targets = self._defensive_target(pseudo_index[selected])
        return F.cross_entropy(logits, targets)


class REFINE_PDB(_PDBMixin, REFINE):
    """REFINE with proactive defensive trigger guidance (PDB-style signal)."""

    def __init__(
        self,
        unet,
        model,
        pretrain=None,
        arr_path=None,
        num_classes=10,
        lmd=0.1,
        seed=0,
        deterministic=False,
        pdb_trigger_type=1,
        pdb_pix_value=1.0,
        pdb_target_shift=1,
        pdb_weight=0.5,
        pdb_batch_ratio=0.5,
        pdb_apply_inference_trigger=True,
        pdb_warmup_ratio=0.3,
        ssl_warmup_ratio=0.3,
        aux_loss_cap_ratio=1.5,
    ):
        super(REFINE_PDB, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic,
        )
        self._init_pdb(
            pdb_trigger_type=pdb_trigger_type,
            pdb_pix_value=pdb_pix_value,
            pdb_target_shift=pdb_target_shift,
            pdb_weight=pdb_weight,
            pdb_batch_ratio=pdb_batch_ratio,
            pdb_apply_inference_trigger=pdb_apply_inference_trigger,
            pdb_warmup_ratio=pdb_warmup_ratio,
            ssl_warmup_ratio=ssl_warmup_ratio,
            aux_loss_cap_ratio=aux_loss_cap_ratio,
        )

    def forward(self, image):
        self.X_adv = torch.clamp(self.unet(image), 0, 1)
        model_input = self._apply_defensive_trigger(self.X_adv) if self.pdb_apply_inference_trigger else self.X_adv
        self.Y_adv = self.model(model_input)

        probs = F.softmax(self.Y_adv, 1)
        if self.pdb_apply_inference_trigger:
            probs = self._inverse_shift_probs(probs)
        probs = self.label_shuffle(probs)
        return probs

    def _test(
        self,
        dataset,
        device,
        batch_size=16,
        num_workers=8,
        loss_func=torch.nn.BCELoss(reduction='none'),
        supconloss_func=SupConLoss(),
    ):
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            self.unet = self.unet.to(device)
            self.unet.eval()
            self.model = self.model.to(device)

            losses = []
            for batch in test_loader:
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
                # Keep validation selection stable: prioritize clean alignment objective only.
                loss = loss_func(logit, f_label).mean() + self.lmd * supconloss
                losses.append(loss.detach().cpu().view(1))

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
                self.unet = nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
        else:
            device = torch.device('cpu')

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker,
        )

        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='mean')
        supconloss_func = SupConLoss()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),
            schedule['lr'],
            schedule['betas'],
            schedule['eps'],
            schedule['weight_decay'],
            schedule['amsgrad'],
        )

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))

        iteration = 0
        last_time = time.time()
        best_eval_loss = float('inf')
        best_epoch = 0
        best_unet_state = None

        msg = (
            f"Total train samples: {len(train_dataset)}\n"
            f"Total test samples: {len(test_dataset)}\n"
            f"Batch size: {schedule['batch_size']}\n"
            f"iteration every epoch: {len(train_dataset) // schedule['batch_size']}\n"
            f"Initial learning rate: {schedule['lr']}\n"
            f"PDB weight: {self.pdb_weight}, batch_ratio: {self.pdb_batch_ratio}, trigger_type: {self.pdb_trigger_type}, target_shift: {self.pdb_target_shift}\n"
            f"PDB warmup ratio: {self.pdb_warmup_ratio}, aux loss cap ratio: {self.aux_loss_cap_ratio}\n"
        )
        log(msg)

        for i in range(schedule['epochs']):
            if i in schedule['schedule']:
                schedule['lr'] *= schedule['gamma']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule['lr']

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

                pdb_loss = self._pdb_guidance_loss(self.X_adv, f_index)
                ce_loss = loss_func(logit, f_label)
                pdb_weight_now = self._effective_aux_weight(self.pdb_weight, i, schedule['epochs'], self.pdb_warmup_ratio)
                pdb_loss_capped = self._cap_aux_loss(pdb_loss, ce_loss)
                loss = ce_loss + self.lmd * supconloss + pdb_weight_now * pdb_loss_capped

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = (
                        time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime())
                        + f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset)//schedule['batch_size']}, "
                                                    f"lr: {schedule['lr']}, loss: {float(loss)}, ce: {float(ce_loss)}, supcon: {float(supconloss)}, "
                                                    f"pdb: {float(pdb_loss)}, pdb_capped: {float(pdb_loss_capped)}, pdb_w: {pdb_weight_now:.4f}, time: {time.time()-last_time}\n"
                    )
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                loss_value = float(loss.item() if hasattr(loss, 'item') else loss)
                msg = '==========Test result on test dataset==========\n' + \
                      time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) + \
                      f'loss: {loss_value}, time: {time.time()-last_time}\n'
                log(msg)

                if loss_value < best_eval_loss:
                    best_eval_loss = loss_value
                    best_epoch = i + 1
                    best_unet_state = deepcopy(self.unet.state_dict())
                    torch.save(best_unet_state, os.path.join(work_dir, 'best_ckpt.pth'))
                    log(f"[Best] epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")

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

        if best_unet_state is not None:
            self.unet.load_state_dict(best_unet_state)
            self.unet = self.unet.to(device)
            self.unet.eval()
            log(f"Restore best checkpoint from epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")


class REFINE_PDB_SSL(REFINE_PDB):
    """REFINE + proactive trigger guidance + self-supervised contrastive objective."""

    def __init__(
        self,
        unet,
        model,
        pretrain=None,
        arr_path=None,
        num_classes=10,
        lmd=0.1,
        seed=0,
        deterministic=False,
        temperature=0.07,
        selfsup_weight=0.02,
        pdb_trigger_type=1,
        pdb_pix_value=1.0,
        pdb_target_shift=1,
        pdb_weight=0.5,
        pdb_batch_ratio=0.5,
        pdb_apply_inference_trigger=True,
        pdb_warmup_ratio=0.3,
        ssl_warmup_ratio=0.3,
        aux_loss_cap_ratio=1.5,
    ):
        super(REFINE_PDB_SSL, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic,
            pdb_trigger_type=pdb_trigger_type,
            pdb_pix_value=pdb_pix_value,
            pdb_target_shift=pdb_target_shift,
            pdb_weight=pdb_weight,
            pdb_batch_ratio=pdb_batch_ratio,
            pdb_apply_inference_trigger=pdb_apply_inference_trigger,
            pdb_warmup_ratio=pdb_warmup_ratio,
            ssl_warmup_ratio=ssl_warmup_ratio,
            aux_loss_cap_ratio=aux_loss_cap_ratio,
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
                worker_init_fn=self._seed_worker,
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
                # Keep validation selection stable: use clean alignment objective only.
                total_loss = ce_loss

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
                self.unet = nn.DataParallel(self.unet.cuda(), device_ids=gpus, output_device=gpus[0])
        else:
            device = torch.device('cpu')

        train_loader = DataLoader(
            train_dataset,
            batch_size=schedule['batch_size'],
            shuffle=True,
            num_workers=schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker,
        )

        self.unet = self.unet.to(device)
        self.unet.train()
        self.model = self.model.to(device)

        loss_func = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            self.unet.parameters(),
            schedule['lr'],
            schedule['betas'],
            schedule['eps'],
            schedule['weight_decay'],
            schedule['amsgrad'],
        )

        work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        torch.save(self.arr_shuffle, os.path.join(work_dir, 'label_shuffle.pth'))

        iteration = 0
        last_time = time.time()
        best_eval_loss = float('inf')
        best_epoch = 0
        best_unet_state = None

        msg = (
            f"Total train samples: {len(train_dataset)}\n"
            f"Total test samples: {len(test_dataset)}\n"
            f"Batch size: {schedule['batch_size']}\n"
            f"iteration every epoch: {len(train_dataset) // schedule['batch_size']}\n"
            f"Initial learning rate: {schedule['lr']}\n"
            f"SSL weight: {self.selfsup_weight}, PDB weight: {self.pdb_weight}, PDB batch_ratio: {self.pdb_batch_ratio}\n"
            f"SSL warmup ratio: {self.ssl_warmup_ratio}, PDB warmup ratio: {self.pdb_warmup_ratio}, aux loss cap ratio: {self.aux_loss_cap_ratio}\n"
        )
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
                ce_loss = loss_func(logit, f_label)
                loss_self = self._selfsup_contrastive_loss(batch_img)
                loss_pdb = self._pdb_guidance_loss(self.X_adv, f_index)
                ssl_weight_now = self._effective_aux_weight(self.selfsup_weight, i, schedule['epochs'], self.ssl_warmup_ratio)
                pdb_weight_now = self._effective_aux_weight(self.pdb_weight, i, schedule['epochs'], self.pdb_warmup_ratio)
                loss_self_capped = self._cap_aux_loss(loss_self, ce_loss)
                loss_pdb_capped = self._cap_aux_loss(loss_pdb, ce_loss)
                total_loss = ce_loss + ssl_weight_now * loss_self_capped + pdb_weight_now * loss_pdb_capped

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    msg = (
                        time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime())
                        + f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset)//schedule['batch_size']}, "
                                                    f"lr: {schedule['lr']}, loss: {float(total_loss)}, ce: {float(ce_loss)}, ssl: {float(loss_self)}, "
                                                    f"ssl_capped: {float(loss_self_capped)}, ssl_w: {ssl_weight_now:.4f}, pdb: {float(loss_pdb)}, "
                                                    f"pdb_capped: {float(loss_pdb_capped)}, pdb_w: {pdb_weight_now:.4f}, time: {time.time()-last_time}\n"
                    )
                    last_time = time.time()
                    log(msg)

            if (i + 1) % schedule['test_epoch_interval'] == 0:
                loss = self._test(test_dataset, device, schedule['batch_size'], schedule['num_workers'])
                loss_value = float(loss.item() if hasattr(loss, 'item') else loss)
                msg = '==========Test result on test dataset==========\n' + \
                      time.strftime('[%Y-%m-%d_%H:%M:%S] ', time.localtime()) + \
                      f'loss: {loss_value}, time: {time.time()-last_time}\n'
                log(msg)

                if loss_value < best_eval_loss:
                    best_eval_loss = loss_value
                    best_epoch = i + 1
                    best_unet_state = deepcopy(self.unet.state_dict())
                    torch.save(best_unet_state, os.path.join(work_dir, 'best_ckpt.pth'))
                    log(f"[Best] epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")

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

        if best_unet_state is not None:
            self.unet.load_state_dict(best_unet_state)
            self.unet = self.unet.to(device)
            self.unet.eval()
            log(f"Restore best checkpoint from epoch={best_epoch}, test_loss={best_eval_loss:.6f}\n")