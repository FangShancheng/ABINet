import torch
import torch.nn as nn

from utils import CharsetMapper


_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048, # 1024
                          dropout=0.1, activation='relu')

class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_length = config.dataset_max_length + 1
        self.charset = CharsetMapper(config.dataset_charset_path, max_length=self.max_length)
    
    def load(self, source, device=None, strict=True):
        state = torch.load(source, map_location=device)
        self.load_state_dict(state['model'], strict=strict)

    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.charset.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_square_subsequent_mask(sz, device, diagonal=0, fw=True):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=diagonal) == 1)
        if fw: mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask
