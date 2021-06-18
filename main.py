import argparse
import logging
import os
import random

import torch
from fastai.callbacks.general_sched import GeneralScheduler, TrainingPhase
from fastai.distributed import *
from fastai.vision import *
from torch.backends import cudnn

from callbacks import DumpPrediction, IterationCallback, TextAccuracy, TopKTextAccuracy
from dataset import ImageDataset, TextDataset
from losses import MultiLosses
from utils import Config, Logger, MyDataParallel, MyConcatDataset


def _set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')

def _get_training_phases(config, n):
    lr = np.array(config.optimizer_lr)
    periods = config.optimizer_scheduler_periods
    sigma = [config.optimizer_scheduler_gamma ** i for i in range(len(periods))]
    phases = [TrainingPhase(n * periods[i]).schedule_hp('lr', lr * sigma[i])
                for i in range(len(periods))]
    return phases

def _get_dataset(ds_type, paths, is_training, config, **kwargs):
    kwargs.update({
        'img_h': config.dataset_image_height,
        'img_w': config.dataset_image_width,
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'data_aug': config.dataset_data_aug,
        'deteriorate_ratio': config.dataset_deteriorate_ratio,
        'is_training': is_training,
        'multiscales': config.dataset_multiscales,
        'one_hot_y': config.dataset_one_hot_y,
    })
    datasets = [ds_type(p, **kwargs) for p in paths]
    if len(datasets) > 1: return MyConcatDataset(datasets)
    else: return datasets[0]


def _get_language_databaunch(config):
    kwargs = {
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'smooth_label': config.dataset_smooth_label,
        'smooth_factor': config.dataset_smooth_factor,
        'one_hot_y': config.dataset_one_hot_y,
        'use_sm': config.dataset_use_sm,
    }
    train_ds = TextDataset(config.dataset_train_roots[0], is_training=True, **kwargs)
    valid_ds = TextDataset(config.dataset_test_roots[0], is_training=False, **kwargs)
    data = DataBunch.create(
        path=train_ds.path,
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_num_workers,
        pin_memory=config.dataset_pin_memory)
    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')
    return data

def _get_databaunch(config):
    # An awkward way to reduce loadding data time during test
    if config.global_phase == 'test': config.dataset_train_roots = config.dataset_test_roots
    train_ds = _get_dataset(ImageDataset, config.dataset_train_roots, True, config)
    valid_ds = _get_dataset(ImageDataset, config.dataset_test_roots, False, config)
    data = ImageDataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_num_workers,
        pin_memory=config.dataset_pin_memory).normalize(imagenet_stats)
    ar_tfm = lambda x: ((x[0], x[1]), x[1])  # auto-regression only for dtd
    data.add_tfm(ar_tfm)

    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')
    
    return data

def _get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    return model


def _get_learner(config, data, model, local_rank=None):
    strict = ifnone(config.model_strict, True)
    if config.global_stage == 'pretrain-language':
        metrics = [TopKTextAccuracy(
            k=ifnone(config.model_k, 5),
            charset_path=config.dataset_charset_path,
            max_length=config.dataset_max_length + 1,
            case_sensitive=config.dataset_eval_case_sensisitves,
            model_eval=config.model_eval)] 
    else:
        metrics = [TextAccuracy(
            charset_path=config.dataset_charset_path,
            max_length=config.dataset_max_length + 1,
            case_sensitive=config.dataset_eval_case_sensisitves,
            model_eval=config.model_eval)]
    opt_type = getattr(torch.optim, config.optimizer_type)
    learner = Learner(data, model, silent=True, model_dir='.',
        true_wd=config.optimizer_true_wd, 
        wd=config.optimizer_wd,
        bn_wd=config.optimizer_bn_wd,
        path=config.global_workdir,
        metrics=metrics,
        opt_func=partial(opt_type, **config.optimizer_args or dict()), 
        loss_func=MultiLosses(one_hot=config.dataset_one_hot_y))
    learner.split(lambda m: children(m))

    if config.global_phase == 'train':
        num_replicas = 1 if local_rank is None else torch.distributed.get_world_size()
        phases = _get_training_phases(config, len(learner.data.train_dl)//num_replicas)
        learner.callback_fns += [
            partial(GeneralScheduler, phases=phases),
            partial(GradientClipping, clip=config.optimizer_clip_grad),
            partial(IterationCallback, name=config.global_name,
                    show_iters=config.training_show_iters,
                    eval_iters=config.training_eval_iters,
                    save_iters=config.training_save_iters,
                    start_iters=config.training_start_iters,
                    stats_iters=config.training_stats_iters)]
    else:
        learner.callbacks += [
            DumpPrediction(learn=learner,
                    dataset='-'.join([Path(p).name for p in config.dataset_test_roots]),charset_path=config.dataset_charset_path,
                    model_eval=config.model_eval,
                    debug=config.global_debug,
                    image_only=config.global_image_only)]

    learner.rank = local_rank
    if local_rank is not None:
        logging.info(f'Set model to distributed with rank {local_rank}.')
        learner.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learner.model)
        learner.model.to(local_rank)
        learner = learner.to_distributed(local_rank)

    if torch.cuda.device_count() > 1 and local_rank is None:
        logging.info(f'Use {torch.cuda.device_count()} GPUs.')
        learner.model = MyDataParallel(learner.model)

    if config.model_checkpoint:
        if Path(config.model_checkpoint).exists():
            with open(config.model_checkpoint, 'rb') as f:
                buffer = io.BytesIO(f.read())
            learner.load(buffer, strict=strict)
        else:
            from distutils.dir_util import copy_tree
            src = Path('/data/fangsc/model')/config.global_name
            trg = Path('/output')/config.global_name
            if src.exists(): copy_tree(str(src), str(trg))
            learner.load(config.model_checkpoint, strict=strict)
        logging.info(f'Read model from {config.model_checkpoint}')
    elif config.global_phase == 'test':
        learner.load(f'best-{config.global_name}', strict=strict)
        logging.info(f'Read model from best-{config.global_name}')

    if learner.opt_func.func.__name__ == 'Adadelta':    # fastai bug, fix after 1.0.60
        learner.fit(epochs=0, lr=config.optimizer_lr)
        learner.opt.mom = 0.

    return learner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--phase', type=str, default=None, choices=['train', 'test'])
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_root', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--image_only', action='store_true', default=None)
    parser.add_argument('--model_strict', action='store_false', default=None)
    parser.add_argument('--model_eval', type=str, default=None, 
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)
    if args.name is not None: config.global_name = args.name
    if args.phase is not None: config.global_phase = args.phase
    if args.test_root is not None: config.dataset_test_roots = [args.test_root]
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.debug is not None: config.global_debug = args.debug
    if args.image_only is not None: config.global_image_only = args.image_only
    if args.model_eval is not None: config.model_eval = args.model_eval
    if args.model_strict is not None: config.model_strict = args.model_strict

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    if args.local_rank is not None:
        logging.info(f'Init distribution training at device {args.local_rank}.')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logging.info('Construct dataset.')
    if config.global_stage == 'pretrain-language': data = _get_language_databaunch(config)
    else: data = _get_databaunch(config)

    logging.info('Construct model.')
    model = _get_model(config)

    logging.info('Construct learner.')
    learner = _get_learner(config, data, model, args.local_rank)

    if config.global_phase == 'train':
        logging.info('Start training.')
        learner.fit(epochs=config.training_epochs,
                    lr=config.optimizer_lr)
    else:
        logging.info('Start validate')
        last_metrics = learner.validate()
        log_str = f'eval loss = {last_metrics[0]:6.3f},  ' \
                  f'ccr = {last_metrics[1]:6.3f},  cwr = {last_metrics[2]:6.3f},  ' \
                  f'ted = {last_metrics[3]:6.3f},  ned = {last_metrics[4]:6.0f},  ' \
                  f'ted/w = {last_metrics[5]:6.3f}, '
        logging.info(log_str)

if __name__ == '__main__':
    main()
