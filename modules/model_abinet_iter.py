import torch
import torch.nn.functional as F
from fastai.vision import *

from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_alignment import BaseAlignment


class ABINetIterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.export = config.export

    def forward(self, images, *args):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
        if self.export:
            return F.softmax(a_res['logits'], dim=2), a_res['pt_lengths']
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res
