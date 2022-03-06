import torch
from torchvision import transforms

label_dic = {0: "NORMAL", 1: "FGR"}

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)

cpu = torch.load(
    "model/densenet121-pretrained-adam-0.0001lr-16bs-25e-80.0val-7e.pth",
    map_location=torch.device("cpu"),
)
cpu.eval()


def inference(imgList: list):
    img = [transform(i) for i in imgList]
    img = torch.stack(img)
    with torch.no_grad():
        outputs = cpu(img)
    prob = [torch.sigmoid(output) for output in outputs]
    res = [label_dic[(p > 0.5).float().item()] for p in prob]
    out = [
        f"[{i[0]}], probability of getting FGR = {100*i[1].item():.2f}%"
        for i in list(zip(res, prob))
    ]
    return out
