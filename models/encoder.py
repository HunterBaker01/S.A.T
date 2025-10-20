from torchvision.models import vgg16
from torch import nn

class MyVGG16(nn.Module):
    def __init__(self, freeze_features=True):
        super().__init__()
        model = vgg16(pretrained=True)

        self.feature= model.features

        if freeze_features:
            for param in self.feature.parameters():
                param.requires_grad = False
    def forward(self, x):
        x = self.feature(x)
        return x
