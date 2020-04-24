import torch
import torch.nn as nn
import torch.nn.functional as F


def _gram_matrix(featmap):
    _, c, h, w = featmap.size()
    gram = torch.matmul(featmap.view(c, h * w), featmap.view(c, h * w).T)
    return gram


class StyleLoss(nn.Module):

    def __init__(self, layer_weights=None):
        super().__init__()
        self.layer_weights = layer_weights

    def load_style_featmaps(self, style_featmaps):
        self.style_featmaps = style_featmaps

    def forward(self, output):
        style_loss = 0
        for i in range(len(output)):
            output_gram = _gram_matrix(output[i])
            style_gram = _gram_matrix(self.style_featmaps[i])
            _, c, h, w = output[i].size()
            layer_loss = F.mse_loss(output_gram, style_gram)\
                / (4 * ((h * w) ** 2))
            if self.layer_weights is not None:
                layer_loss *= self.layer_weights[i]
            style_loss += layer_loss

        return style_loss

class ContentLoss(nn.Module):

    def __init__(self, layer_weights=None):
        super().__init__()
        self.layer_weights = layer_weights

    def load_content_featmaps(self, content_featmaps):
        self.content_featmaps = content_featmaps

    def forward(self, output):
        content_loss = 0
        for i in range(len(output)):
            if self.layer_weights is not None:
                content_loss += F.mse_loss(output[i],
                    self.content_featmaps[i]) * self.layer_weights[i]
            else:
                content_loss += F.mse_loss(output[i],
                    self.content_featmaps[i])

        return content_loss


class VariationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image):
        n, h, w, c = image.size()
        x_variation = image[:, :, :, :-1] - image[:, :, :, 1:]
        y_variation = image[:, :, :-1, :] - image[:, :, 1:, :]
        var_loss = torch.sqrt(torch.pow(x_variation, 2).mean()
            + torch.pow(y_variation, 2).mean())

        return var_loss


class TotalLoss(nn.Module):

    def __init__(self, style_layer_weights=None,
            content_layer_weights=None, style_loss_weight=1.0,
            content_loss_weight=10.0, variation_loss_weight=1.0):
        super().__init__()
        self.style_loss = StyleLoss(style_layer_weights)
        self.content_loss = ContentLoss(content_layer_weights)
        self.variation_loss = VariationLoss()
        self.style_loss_weight = style_loss_weight
        self.content_loss_weight = content_loss_weight
        self.variation_loss_weight = variation_loss_weight

    def load_featmaps(self, style_featmaps, content_featmaps):
        self.style_loss.load_style_featmaps(style_featmaps)
        self.content_loss.load_content_featmaps(content_featmaps)

    def forward(self, style_output, content_output, input_img):
        style_loss_val = self.style_loss(style_output)\
            * self.style_loss_weight
        content_loss_val = self.content_loss(content_output)\
            * self.content_loss_weight
        variation_loss_val = self.variation_loss(input_img)

        total_loss_val = style_loss_val + content_loss_val\
            + variation_loss_val

        return style_loss_val, content_loss_val, total_loss_val
