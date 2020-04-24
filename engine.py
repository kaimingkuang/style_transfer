import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_image


class Engine(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = {"style": [], "content": [], "total": []}

    def compile(self, model, loss_fn, optimizer, style_img, content_img):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.style_img = style_img
        self.content_img = content_img
        self.style_target, _ = self.model(style_img)
        _, self.content_target = self.model(content_img)
        self.loss_fn.load_featmaps([x.detach() for x in self.style_target],
            [x.detach() for x in self.content_target])

    def fit(self, input_img):
        progress = tqdm(total=self.cfg.epochs)
        save_image(input_img, os.path.join(self.cfg.save_path,
            "input_epoch_0.jpg"))

        for i in range(self.cfg.epochs):
            def closure():
                self.optimizer.zero_grad()
                input_img.data.clamp_(0, 255)

                style_output, content_output = self.model(input_img)
                style_loss, content_loss, total_loss = self.loss_fn(
                    style_output, content_output, input_img)
                style_loss_val = style_loss.detach().item()
                content_loss_val = content_loss.detach().item()
                progress.set_postfix_str(
                    "style_loss={:.4f}, content_loss={:.4f}"\
                    .format(style_loss_val, content_loss_val))
                loss_dict = {"style": style_loss_val,
                    "content": content_loss_val,
                    "total": total_loss.detach().item()}
                self._log_history(loss_dict)
                total_loss.backward(retain_graph=True)

                return total_loss
            self.optimizer.step(closure)
            if (i + 1) % 5 == 0:
                save_image(input_img, os.path.join(self.cfg.save_path,
                    "input_epoch_{}.jpg".format(i + 1)))
            progress.update(1)

        input_img.data.clamp_(0, 255)
        save_image(input_img, os.path.join(self.cfg.save_path, "output.jpg"))

    def _log_history(self, loss_dict):
        for k in loss_dict.keys():
            if k in self.history.keys():
                self.history[k].append(loss_dict[k])
            else:
                raise ValueError("Unrecognized loss key {}".format(k))
