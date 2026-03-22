
'''
REFINE-CG: Confidence-Gated variant of REFINE.
This file keeps the same external interface as REFINE and only extends inference behavior.
'''

import numpy as np
import torch
import torch.nn.functional as F

from .REFINE import REFINE


class REFINE_CG(REFINE):
    """Confidence-Gated REFINE.

    The constructor and public methods remain compatible with REFINE.
    Additional optional args control the gate during inference.

    Args:
        cg_threshold (float): Suspicion threshold for gate activation.
        cg_temperature (float): Temperature for gate sharpness.
        cg_strength (float): Multiplicative strength for gate in [0, +inf).
        cg_enable (bool): Whether to enable gated inference.
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
                 cg_threshold=0.35,
                 cg_temperature=0.10,
                 cg_strength=1.0,
                 cg_enable=True):
        super(REFINE_CG, self).__init__(
            unet=unet,
            model=model,
            pretrain=pretrain,
            arr_path=arr_path,
            num_classes=num_classes,
            lmd=lmd,
            seed=seed,
            deterministic=deterministic,
        )
        self.cg_threshold = float(cg_threshold)
        self.cg_temperature = max(float(cg_temperature), 1e-6)
        self.cg_strength = float(cg_strength)
        self.cg_enable = bool(cg_enable)

        # cached for optional analysis
        self.last_suspicion_score = None
        self.last_gate = None

    def _safe_label_shuffle(self, probs):
        """Device-safe label shuffle equivalent to REFINE.label_shuffle."""
        label_new = torch.zeros_like(probs)
        index = torch.from_numpy(self.arr_shuffle).repeat(probs.shape[0], 1).to(probs.device)
        return label_new.scatter(1, index, probs)

    @staticmethod
    def _js_divergence(p, q, eps=1e-8):
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        m = 0.5 * (p + q)
        js = 0.5 * (torch.sum(p * (torch.log(p) - torch.log(m)), dim=1) +
                    torch.sum(q * (torch.log(q) - torch.log(m)), dim=1))
        return js

    def forward(self, image):
        # REFINE branch
        self.X_adv = torch.clamp(self.unet(image), 0, 1)
        self.Y_adv = self.model(self.X_adv)
        refine_probs = F.softmax(self.Y_adv, 1)
        refine_probs = self._safe_label_shuffle(refine_probs)

        if not self.cg_enable:
            return refine_probs

        # Clean branch for confidence gating
        clean_logits = self.model(image)
        clean_probs = F.softmax(clean_logits, 1)
        clean_probs = self._safe_label_shuffle(clean_probs)

        clean_conf = clean_probs.max(dim=1).values
        js = self._js_divergence(clean_probs, refine_probs)

        # Higher score => more suspicious => prefer stronger refinement branch.
        suspicion_score = torch.clamp((1.0 - clean_conf) + js, 0.0, 1.0)
        gate = torch.sigmoid((suspicion_score - self.cg_threshold) / self.cg_temperature)
        gate = torch.clamp(gate * self.cg_strength, 0.0, 1.0)

        self.last_suspicion_score = suspicion_score.detach().cpu()
        self.last_gate = gate.detach().cpu()

        gate = gate.unsqueeze(1)
        return gate * refine_probs + (1.0 - gate) * clean_probs

    def train_unet(self, train_dataset, test_dataset, schedule):
        # Keep training behavior aligned with vanilla REFINE.
        old_cg_enable = self.cg_enable
        self.cg_enable = False
        try:
            super(REFINE_CG, self).train_unet(train_dataset, test_dataset, schedule)
        finally:
            self.cg_enable = old_cg_enable
