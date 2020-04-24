import torch
from torch import optim

from losses import TotalLoss


# pretrained model path
pretrained_path = "pretrained_model/vgg19.pth"

# image config
target_size = (512, 512)
img_mean = torch.tensor([0.485, 0.456, 0.406])
img_std = torch.tensor([0.229, 0.224, 0.225])
save_path = "images/output"
init_img = "content"

# train config
optimizer = optim.LBFGS
epochs = 100
init_lr = 0.5
log_path = "logs"
scheduler = None

loss_fn = TotalLoss
style_layers = ["block_{}_conv_1".format(i) for i in range(5)]
content_layers = ["block_4_conv_1"]
style_layer_weights = [1 / len(style_layers)
    for i in range(len(style_layers))]
content_layer_weights = None
style_loss_weight = 1000000.0
content_loss_weight = 1.0

style_img_path = "images/style"
content_img_path = "images/content"
