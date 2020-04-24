import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class NormalizationLayer(nn.Module):

    def __init__(self, img_mean, img_std):
        super().__init__()
        self.img_mean = img_mean.view(-1, 1, 1)
        self.img_std = img_std.view(-1, 1, 1)

    def forward(self, input_image):
        return (input_image / 255.0 - self.img_mean) / self.img_std


class GatysModel(nn.Module):

    def __init__(self, style_layers, content_layers, img_mean, img_std):
        super().__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        vgg = vgg19(pretrained=True)
        self.add_module("norm", NormalizationLayer(img_mean, img_std))
        i = 0
        j = 0
        for _, module in vgg.features.named_children():
            if isinstance(module, nn.Conv2d):
                name = "block_{}_conv_{}".format(i, j)
            elif isinstance(module, nn.ReLU):
                name = "block_{}_relu_{}".format(i, j)
                j += 1
            elif isinstance(module, nn.MaxPool2d):
                name = "block_{}_maxpool".format(i)
                i += 1
                j = 0
            else:
                raise Exception("Unrecognized module.")

            self.add_module(name, module)

            if "conv" in name and name >= style_layers[-1]\
                    and name >= content_layers[-1]:
                break

    def forward(self, input_image):
        self.eval()
        style_output = []
        content_output = []
        x = input_image
        for name, module in self.named_children():
            x = module(x)
            if name in self.style_layers:
                style_output.append(x)
            if name in self.content_layers:
                content_output.append(x)

        return style_output, content_output
