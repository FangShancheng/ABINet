import torch
import torch.nn as nn
from fastai.vision import *

from modules.model import Model, _default_tfmer_cfg


class BaseAlignment(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}
