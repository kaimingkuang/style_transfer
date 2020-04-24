import argparse
import os
from copy import deepcopy

import cv2
import torch
from matplotlib import pyplot as plt

from engine import Engine
from model import GatysModel
from utils import generate_random_image, preprocess_image, read_image


def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", help="The path of the style image.",
        default="style.jpg")
    parser.add_argument("--content", help="The path of the content image.",
        default="content.jpg")
    args = parser.parse_args()

    style_img = preprocess_image(read_image(os.path.join(cfg.style_img_path,
        args.style)), cfg.target_size)
    content_img = preprocess_image(read_image(os.path.join(
        cfg.content_img_path, args.content)), cfg.target_size)

    if cfg.init_img == "content":
        input_img = deepcopy(content_img)
    elif cfg.init_img == "style":
        input_img = deepcopy(style_img)
    else:
        input_img = generate_random_image(content_img)

    model = GatysModel(cfg.pretrained_path, cfg.style_layers,
        cfg.content_layers, cfg.img_mean, cfg.img_std)

    if torch.cuda.is_available():
        style_img = style_img.cuda()
        content_img = content_img.cuda()
        input_img = input_img.cuda()
        model = GatysModel(cfg.pretrained_path, cfg.style_layers,
            cfg.content_layers, cfg.img_mean.cuda(),
            cfg.img_std.cuda()).cuda()
    else:
        model = GatysModel(cfg.pretrained_path, cfg.style_layers,
            cfg.content_layers, cfg.img_mean, cfg.img_std)

    engine = Engine(cfg)

    loss_fn = cfg.loss_fn(cfg.style_layer_weights, cfg.content_layer_weights,
        cfg.style_loss_weight, cfg.content_loss_weight)
    optimizer = cfg.optimizer([input_img.requires_grad_()], lr=cfg.init_lr)
    if cfg.scheduler is not None:
        scheduler = cfg.scheduler(optimizer, **cfg.params_scheduler)
    else:
        scheduler = cfg.scheduler

    engine.compile(model, loss_fn, optimizer, style_img, content_img)
    engine.fit(input_img, scheduler)


if __name__ == "__main__":
    import config


    main(config)
