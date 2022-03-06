from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1
from torchvision.models.mobilenet import (
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)
from torchvision.models.inception import inception_v3
from torchvision.models.densenet import densenet121, densenet169
from torchvision.models.vgg import vgg16_bn, vgg19_bn
import torch.nn as nn


def get_model(model_name):
    split = model_name.split("-")
    if split[0] == "resnet18":
        model = (
            resnet18(pretrained=True)
            if split[1] == "pretrained"
            else resnet18(pretrained=False)
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif split[0] == "resnet50":
        model = (
            resnet50(pretrained=True)
            if split[1] == "pretrained"
            else resnet50(pretrained=False)
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif split[0] == "efficientnet_b0":
        model = (
            efficientnet_b0(pretrained=True)
            if split[1] == "pretrained"
            else efficientnet_b0(pretrained=False)
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif split[0] == "efficientnet_b1":
        model = (
            efficientnet_b1(pretrained=True)
            if split[1] == "pretrained"
            else efficientnet_b1(pretrained=False)
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif split[0] == "vgg16_bn":
        model = (
            vgg16_bn(pretrained=True)
            if split[1] == "pretrained"
            else vgg16_bn(pretrained=False)
        )
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)

    elif split[0] == "vgg19_bn":
        model = (
            vgg19_bn(pretrained=True)
            if split[1] == "pretrained"
            else vgg19_bn(pretrained=False)
        )
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)

    elif split[0] == "densenet121":
        model = (
            densenet121(pretrained=True)
            if split[1] == "pretrained"
            else densenet121(pretrained=False)
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)

    elif split[0] == "densenet169":
        model = (
            densenet169(pretrained=True)
            if split[1] == "pretrained"
            else densenet169(pretrained=False)
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)

    elif split[0] == "mobilenet_v2":
        model = (
            mobilenet_v2(pretrained=True)
            if split[1] == "pretrained"
            else mobilenet_v2(pretrained=False)
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif split[0] == "mobilenet_v3_small":
        model = (
            mobilenet_v3_small(pretrained=True)
            if split[1] == "pretrained"
            else mobilenet_v3_small(pretrained=False)
        )
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)

    elif split[0] == "mobilenet_v3_large":
        model = (
            mobilenet_v3_large(pretrained=True)
            if split[1] == "pretrained"
            else mobilenet_v3_large(pretrained=False)
        )
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)

    else:
        model = None

    return model
