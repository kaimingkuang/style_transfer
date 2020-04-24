import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor


def read_image(path):
    img = plt.imread(path)
    return img


def preprocess_image(img, target_size):
    img = cv2.resize(img, target_size)
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    return img


def save_image(img_tensor, path):
    img_array = np.transpose(np.squeeze(
        img_tensor.detach().cpu().numpy(), axis=0), (1, 2, 0))
    cv2.imwrite(path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


def generate_random_image(template_img):
    return torch.rand_like(template_img) * 255
