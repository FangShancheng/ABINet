import argparse
import logging
import os
import glob
import tqdm
import torch
import PIL
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import Config, Logger, CharsetMapper

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='figs/test')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='workdir/train-abinet/best-train-abinet.pth')
    parser.add_argument('--model_eval', type=str, default='alignment', 
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)

    logging.info('Construct model.')
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)

    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, fname) for fname in os.listdir(args.input)]
    else:
        paths = glob.glob(os.path.expanduser(args.input))
        assert paths, "The input path(s) was not found"
    paths = sorted(paths)
    for path in tqdm.tqdm(paths):
        img = PIL.Image.open(path).convert('RGB')
        img = preprocess(img, config.dataset_image_width, config.dataset_image_height)
        img = img.to(device)
        res = model(img)
        pt_text, _, __ = postprocess(res, charset, config.model_eval)
        logging.info(f'{path}: {pt_text[0]}')

if __name__ == '__main__':
    main()
