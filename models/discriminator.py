import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def _drop_cls_and_reshape(tokens: torch.Tensor) -> torch.Tensor:
    # tokens: [B, N, D] with CLS at index 0
    B, N, D = tokens.shape
    if int(math.isqrt(N)) ** 2 != N:  # not square -> drop CLS
        tokens = tokens[:, 1:, :]
        N = tokens.shape[1]
    H = W = int(math.isqrt(N))
    assert H * W == N, f"Token count {N} is not square after CLS drop."
    return tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]

class SARAResNet18Discriminator(nn.Module):
    """
    1x1 conv projects D->3. Backbone = ResNet18 stem+layer1+layer2 (pretrained).
    GAP + linear -> 1 logit.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.tokens_to_img = nn.Conv2d(in_dim, 3, kernel_size=1)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.body = nn.Sequential(backbone.layer1, backbone.layer2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone.layer2[-1].conv2.out_channels, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = _drop_cls_and_reshape(tokens)  # drop CLS for adv alignment
        x = self.tokens_to_img(x)
        x = self.stem(x)
        x = self.body(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)  # [B,1]

def build_sara_discriminator(in_dim: int, device: torch.device) -> nn.Module:
    return SARAResNet18Discriminator(in_dim).to(device)
