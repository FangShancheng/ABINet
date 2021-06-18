import logging
import shutil
import time

import editdistance as ed
import torchvision.utils as vutils
from fastai.callbacks.tensorboard import (LearnerTensorboardWriter,
                                          SummaryWriter, TBWriteRequest,
                                          asyncTBWriter)
from fastai.vision import *
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

import dataset
from utils import CharsetMapper, Timer, blend_mask


class IterationCallback(LearnerTensorboardWriter):
    "A `TrackerCallback` that monitor in each iteration."
    def __init__(self, learn:Learner, name:str='model', checpoint_keep_num=5,
                 show_iters:int=50, eval_iters:int=1000, save_iters:int=20000,
                 start_iters:int=0, stats_iters=20000):
        #if self.learn.rank is not None: time.sleep(self.learn.rank)  # keep all event files
        super().__init__(learn, base_dir='.', name=learn.path, loss_iters=show_iters, 
                        stats_iters=stats_iters, hist_iters=stats_iters)
        self.name, self.bestname = Path(name).name, f'best-{Path(name).name}'
        self.show_iters = show_iters
        self.eval_iters = eval_iters
        self.save_iters = save_iters
        self.start_iters = start_iters
        self.checpoint_keep_num = checpoint_keep_num
        self.metrics_root = 'metrics/'  # rewrite
        self.timer = Timer()
        self.host = self.learn.rank is None or self.learn.rank == 0

    def _write_metrics(self, iteration:int, names:List[str], last_metrics:MetricsList)->None:
        "Writes training metrics to Tensorboard."
        for i, name in enumerate(names):
            if last_metrics is None or len(last_metrics) < i+1: return
            scalar_value = last_metrics[i]
            self._write_scalar(name=name, scalar_value=scalar_value, iteration=iteration)

    def _write_sub_loss(self, iteration:int, last_losses:dict)->None:
        "Writes sub loss to Tensorboard."
        for name, loss in last_losses.items():
            scalar_value = to_np(loss)
            tag = self.metrics_root + name
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _save(self, name):
        if isinstance(self.learn.model, DistributedDataParallel):
            tmp = self.learn.model
            self.learn.model = self.learn.model.module
            self.learn.save(name)
            self.learn.model = tmp
        else: self.learn.save(name)

    def _validate(self, dl=None, callbacks=None, metrics=None, keeped_items=False):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.learn.data.valid_dl)
        metrics = ifnone(metrics, self.learn.metrics)
        cb_handler = CallbackHandler(ifnone(callbacks, []), metrics)
        cb_handler.on_train_begin(1, None, metrics); cb_handler.on_epoch_begin()
        if keeped_items: cb_handler.state_dict.update(dict(keeped_items=[]))
        val_metrics = validate(self.learn.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        if keeped_items: return cb_handler.state_dict['keeped_items']
        else: return cb_handler.state_dict['last_metrics']

    def jump_to_epoch_iter(self, epoch:int, iteration:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch}_{iteration}', purge=False)
            logging.info(f'Loaded {self.name}_{epoch}_{iteration}')
        except: logging.info(f'Model {self.name}_{epoch}_{iteration} not found.')

    def on_train_begin(self, n_epochs, **kwargs):
        # TODO: can not write graph here
        # super().on_train_begin(**kwargs)
        self.best = -float('inf')
        self.timer.tic()
        if self.host:
            checkpoint_path = self.learn.path/'checkpoint.yaml'
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
            open(checkpoint_path, 'w').close()
        return {'skip_validate': True, 'iteration':self.start_iters}  # disable default validate

    def on_batch_begin(self, **kwargs:Any)->None:
        self.timer.toc_data()
        super().on_batch_begin(**kwargs)

    def on_batch_end(self, iteration, epoch, last_loss, smooth_loss, train, **kwargs):
        super().on_batch_end(last_loss, iteration, train, **kwargs)
        if iteration == 0: return

        if iteration % self.loss_iters == 0:
            last_losses = self.learn.loss_func.last_losses
            self._write_sub_loss(iteration=iteration, last_losses=last_losses)
            self.tbwriter.add_scalar(tag=self.metrics_root + 'lr',
                scalar_value=self.opt.lr, global_step=iteration)

        if iteration % self.show_iters == 0:
            log_str = f'epoch {epoch} iter {iteration}: loss = {last_loss:6.4f},  ' \
                      f'smooth loss = {smooth_loss:6.4f}'
            logging.info(log_str)
            # log_str = f'data time = {self.timer.data_diff:.4f}s, runing time = {self.timer.running_diff:.4f}s'
            # logging.info(log_str)

        if iteration % self.eval_iters == 0:
            # TODO: or remove time to on_epoch_end
            # 1. Record time 
            log_str = f'average data time = {self.timer.average_data_time():.4f}s, ' \
                      f'average running time = {self.timer.average_running_time():.4f}s'
            logging.info(log_str)

            # 2. Call validate
            last_metrics = self._validate()
            self.learn.model.train()
            log_str = f'epoch {epoch} iter {iteration}: eval loss = {last_metrics[0]:6.4f},  ' \
                      f'ccr = {last_metrics[1]:6.4f},  cwr = {last_metrics[2]:6.4f},  ' \
                      f'ted = {last_metrics[3]:6.4f},  ned = {last_metrics[4]:6.4f},  ' \
                      f'ted/w = {last_metrics[5]:6.4f}, '
            logging.info(log_str)
            names = ['eval_loss', 'ccr', 'cwr', 'ted', 'ned', 'ted/w']
            self._write_metrics(iteration, names, last_metrics)

            # 3. Save best model
            current = last_metrics[2]
            if current is not None and current > self.best:
                logging.info(f'Better model found at epoch {epoch}, '\
                             f'iter {iteration} with accuracy value: {current:6.4f}.')
                self.best = current
                self._save(f'{self.bestname}')

        if iteration % self.save_iters == 0 and self.host:
            logging.info(f'Save model {self.name}_{epoch}_{iteration}')
            filename = f'{self.name}_{epoch}_{iteration}'
            self._save(filename)

            checkpoint_path = self.learn.path/'checkpoint.yaml'
            if not checkpoint_path.exists():
                open(checkpoint_path, 'w').close()
            with open(checkpoint_path, 'r') as file:
                checkpoints = yaml.load(file, Loader=yaml.FullLoader) or dict()
            checkpoints['all_checkpoints'] = (
                checkpoints.get('all_checkpoints') or list())
            checkpoints['all_checkpoints'].insert(0, filename)
            if len(checkpoints['all_checkpoints']) > self.checpoint_keep_num:
                removed_checkpoint = checkpoints['all_checkpoints'].pop()
                removed_checkpoint =  self.learn.path/self.learn.model_dir/f'{removed_checkpoint}.pth'
                os.remove(removed_checkpoint)
            checkpoints['current_checkpoint'] = filename
            with open(checkpoint_path, 'w') as file:
                yaml.dump(checkpoints, file)


        self.timer.toc_running()

    def on_train_end(self, **kwargs):
        #self.learn.load(f'{self.bestname}', purge=False)
        pass
    
    def on_epoch_end(self, last_metrics:MetricsList, iteration:int, **kwargs)->None:
        self._write_embedding(iteration=iteration)


class TextAccuracy(Callback):
    _names = ['ccr', 'cwr', 'ted', 'ned', 'ted/w']
    def __init__(self, charset_path, max_length, case_sensitive, model_eval):
        self.charset_path = charset_path
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.charset = CharsetMapper(charset_path, self.max_length)
        self.names = self._names

        self.model_eval = model_eval or 'alignment'
        assert self.model_eval in ['vision', 'language', 'alignment']
 
    def on_epoch_begin(self, **kwargs):
        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.
        self.total_ed = 0.
        self.total_ned = 0.

    def _get_output(self, last_output):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == self.model_eval: output = res
        else: output = last_output
        return output
    
    def _update_output(self, last_output, items):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == self.model_eval: res.update(items)
        else: last_output.update(items)
        return last_output

    def on_batch_end(self, last_output, last_target, **kwargs):
        output = self._get_output(last_output)
        logits, pt_lengths = output['logits'], output['pt_lengths']
        pt_text, pt_scores, pt_lengths_ = self.decode(logits)
        assert (pt_lengths == pt_lengths_).all(), f'{pt_lengths} != {pt_lengths_} for {pt_text}'
        last_output = self._update_output(last_output, {'pt_text':pt_text, 'pt_scores':pt_scores})

        pt_text = [self.charset.trim(t) for t in pt_text]
        label = last_target[0]
        if label.dim() == 3: label = label.argmax(dim=-1)  # one-hot label
        gt_text = [self.charset.get_text(l, trim=True) for l in label]
        
        for i in range(len(gt_text)):
            if not self.case_sensitive:
                gt_text[i], pt_text[i] = gt_text[i].lower(), pt_text[i].lower()
            distance = ed.eval(gt_text[i], pt_text[i])
            self.total_ed += distance
            self.total_ned += float(distance) / max(len(gt_text[i]), 1)

            if gt_text[i] == pt_text[i]:
                self.correct_num_word += 1
            self.total_num_word += 1
            
            for j in range(min(len(gt_text[i]), len(pt_text[i]))):
                if gt_text[i][j] == pt_text[i][j]:
                    self.correct_num_char += 1
            self.total_num_char += len(gt_text[i])

        return {'last_output': last_output}

    def on_epoch_end(self, last_metrics, **kwargs):
        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                self.total_ed,
                self.total_ned,
                self.total_ed / self.total_num_word]
        return add_metrics(last_metrics, mets)

    def decode(self, logit):
        """ Greed decode """
        # TODO: test running time and decode on GPU
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, self.max_length))  # one for end-token
        pt_scores = torch.stack(pt_scores)
        pt_lengths = pt_scores.new_tensor(pt_lengths, dtype=torch.long)
        return pt_text, pt_scores, pt_lengths


class TopKTextAccuracy(TextAccuracy):
    _names = ['ccr', 'cwr']
    def __init__(self, k, charset_path, max_length, case_sensitive, model_eval):
        self.k = k
        self.charset_path = charset_path
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.charset = CharsetMapper(charset_path, self.max_length)
        self.names = self._names
 
    def on_epoch_begin(self, **kwargs):
        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.

    def on_batch_end(self, last_output, last_target, **kwargs):
        logits, pt_lengths = last_output['logits'], last_output['pt_lengths']
        gt_labels, gt_lengths = last_target[:]

        for logit, pt_length, label, length in zip(logits, pt_lengths, gt_labels, gt_lengths):
            word_flag = True
            for i in range(length):
                char_logit = logit[i].topk(self.k)[1]
                char_label = label[i].argmax(-1)
                if char_label in char_logit: self.correct_num_char += 1
                else: word_flag = False
                self.total_num_char += 1
            if pt_length == length and word_flag:
                self.correct_num_word += 1
            self.total_num_word += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                0., 0., 0.]
        return add_metrics(last_metrics, mets)


class DumpPrediction(LearnerCallback):

    def __init__(self, learn, dataset, charset_path, model_eval, image_only=False, debug=False):
        super().__init__(learn=learn)
        self.debug = debug
        self.model_eval = model_eval or 'alignment'
        self.image_only = image_only
        assert self.model_eval in ['vision', 'language', 'alignment']

        self.dataset, self.root = dataset, Path(self.learn.path)/f'{dataset}-{self.model_eval}'
        self.attn_root = self.root/'attn'
        self.charset = CharsetMapper(charset_path)
        if self.root.exists(): shutil.rmtree(self.root)
        self.root.mkdir(), self.attn_root.mkdir()

        self.pil = transforms.ToPILImage()
        self.tensor = transforms.ToTensor()
        size = self.learn.data.img_h, self.learn.data.img_w
        self.resize = transforms.Resize(size=size, interpolation=0)
        self.c = 0

    def on_batch_end(self, last_input, last_output, last_target, **kwargs):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                if res['name'] == self.model_eval: pt_text = res['pt_text']
                if res['name'] == 'vision': attn_scores = res['attn_scores'].detach().cpu()
                if res['name'] == self.model_eval: logits = res['logits']
        else:
            pt_text = last_output['pt_text']
            attn_scores = last_output['attn_scores'].detach().cpu()
            logits = last_output['logits']

        images = last_input[0] if isinstance(last_input, (tuple, list)) else last_input
        images = images.detach().cpu()
        pt_text = [self.charset.trim(t) for t in pt_text]
        gt_label = last_target[0]
        if gt_label.dim() == 3: gt_label = gt_label.argmax(dim=-1)  # one-hot label
        gt_text = [self.charset.get_text(l, trim=True) for l in gt_label]

        prediction, false_prediction = [], []
        for gt, pt, image, attn, logit in zip(gt_text, pt_text, images, attn_scores, logits):
            prediction.append(f'{gt}\t{pt}\n')
            if gt != pt:
                if self.debug:
                    scores = torch.softmax(logit, dim=-1)[:max(len(pt), len(gt)) + 1]
                    logging.info(f'{self.c} gt {gt}, pt {pt}, logit {logit.shape}, scores {scores.topk(5, dim=-1)}')
                false_prediction.append(f'{gt}\t{pt}\n')

            image = self.learn.data.denorm(image)
            if not self.image_only:
                image_np = np.array(self.pil(image))
                attn_pil = [self.pil(a) for a in attn[:, None, :, :]]
                attn = [self.tensor(self.resize(a)).repeat(3, 1, 1) for a in attn_pil]
                attn_sum = np.array([np.array(a) for a in attn_pil[:len(pt)]]).sum(axis=0)
                blended_sum = self.tensor(blend_mask(image_np, attn_sum))
                blended = [self.tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
                save_image = torch.stack([image] + attn + [blended_sum] + blended)
                save_image = save_image.view(2, -1, *save_image.shape[1:])
                save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)
                vutils.save_image(save_image, self.attn_root/f'{self.c}_{gt}_{pt}.jpg', 
                                nrow=2, normalize=True, scale_each=True)
            else:
                self.pil(image).save(self.attn_root/f'{self.c}_{gt}_{pt}.jpg')
            self.c += 1

        with open(self.root/f'{self.model_eval}.txt', 'a') as f: f.writelines(prediction)
        with open(self.root/f'{self.model_eval}-false.txt', 'a') as f: f.writelines(false_prediction)  
