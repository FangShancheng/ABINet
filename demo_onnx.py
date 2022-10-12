""" Created by MrBBS """
# 10/6/2022
# -*-encoding:utf-8-*-


import onnxruntime
import torch
import cv2
import numpy as np
from torchvision import transforms
from utils import CharsetMapper
from torch.nn import functional as F
from typing import *
import tqdm

sess = onnxruntime.InferenceSession('abinet.onnx', providers=['CUDAExecutionProvider'])

charset = CharsetMapper(filename='data/charset_vn.txt',
                        max_length=51)


class WordAccuary:

    def __init__(self, case_sensitive=False):
        super(WordAccuary, self).__init__()
        self.total = 1e-10
        self.correct = 0
        self.case_sensitive = case_sensitive

    def update(self, y_pred: List[str], y: List[str]):
        self.total += len(y_pred)

        for pred, gt in zip(y_pred, y):
            if not self.case_sensitive:
                pred = pred.lower()
                gt = gt.lower()

            if pred == gt:
                self.correct += 1

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.total = 1e-10
        self.correct = 0


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


def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return (img - mean[..., None, None]) / std[..., None, None]


imgs = r'E:\text_recognize_data\syntext_line'

# cv2.namedWindow('cc', cv2.WINDOW_NORMAL)

with open(r'E:\text_recognize_data\syntext_line\anno_real.txt', 'r', encoding='utf8') as f:
    data = f.readlines()
metric = WordAccuary()

# for i in tqdm.tqdm(data[:4000]):
for i in data:
    img_path, text = i.strip().split('\t')
    image = cv2.imread(imgs + f'/{img_path}')[:, :, ::-1]
    image = preprocess(image, 128, 32)
    image = image.cpu().numpy()

    inp = {'input': image}

    logits, lengths = sess.run(["logits", "lengths"], inp)

    # print(logits)

    pt_text, pt_scores, pt_lengths = _decode(torch.from_numpy(logits))
    metric.update(pt_text[0], text)
    print(text, pt_text[0])
    # break

print(metric.compute())
