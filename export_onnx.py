""" Created by MrBBS """

import os

import torch

from utils import Config


def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    model = model.eval()
    return model


def load(model, file, device=None, strict=True):
    if device is None:
        device = 'cpu'
    elif isinstance(device, int):
        device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='figs/test')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='workdir_bbs/train-abinet/best-train-abinet.pth')
    parser.add_argument('--model_eval', type=str, default='alignment',
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'
    config.export = True

    # Logger.init(config.global_workdir, config.global_name, config.global_phase)
    # Logger.enable_file()
    print(config)

    # logging.info('Construct model.')
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    x = torch.rand((1, 3, config.dataset_image_height, config.dataset_image_width), requires_grad=True)

    torch.onnx.export(model, x, 'abinet.onnx',
                      verbose=True, opset_version=13,
                      do_constant_folding=True,
                      export_params=True,
                      input_names=["input"],
                      output_names=["logits", "lengths"],
                      dynamic_axes={"input": {0: "batch"}})

