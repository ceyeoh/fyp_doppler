import numpy as np
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1
from torchvision.models.mobilenet import (
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)
from torchvision.models.inception import inception_v3
from torchvision.models.densenet import densenet121
from torchvision.models.vgg import vgg16_bn, vgg19_bn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from utils import loader, stratified
import warnings

warnings.filterwarnings("ignore")

# config seed for reproducible codes
SEED = 17167055
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

path = Path("data/preprocess/")
data = [str(i) for i in list(path.glob("*.jpg"))]

# load images
def loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


# stratified sampling
def stratified(df, col, n_smaples):
    n = min(n_smaples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


df = pd.DataFrame()
df["filename"] = [i.split("\\")[-1] for i in data]
df["pixels"] = [loader(i) for i in data]
df["label"] = [int(i[0]) for i in df["filename"]]
df["id"] = ["-".join(i.split("_")[1:-2]) for i in df["filename"]]
df["var"] = [i.split(".")[0][-1] for i in df["filename"]]
df["ga"] = [i.split("_")[-2] for i in df["filename"]]

# remove all duplicates to make dataset more clean
df = df[df["var"] == "0"].reset_index(drop=True)

df_duplicated = df[df.duplicated("id", keep=False)]
df_unique = df.drop(df_duplicated.index)
df_duplicated = df_duplicated.reset_index(drop=True)
df_unique = df_unique.reset_index(drop=True)

# test set
df_test = stratified(df_unique, "label", 5)
df_unique = df_unique.drop(df_test.index).reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# validation set
df_val = stratified(df_unique, "label", 5)
df_unique = df_unique.drop(df_val.index).reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# train set
df_train = pd.concat([df_duplicated, df_unique], ignore_index=True)

# Display label counts for each dataset
print(f"Training set : {df_train.label.count()}\n{df_train.label.value_counts()}\n")
print(f"Validation set : {df_val.label.count()}\n{df_val.label.value_counts()}\n")
print(f"Test set : {df_test.label.count()}\n{df_test.label.value_counts()}\n")

transform = {
    "train": A.Compose(
        [
            A.CLAHE(p=0.8),
            A.RandomCrop(112, 112),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    ),
    "test": A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    ),
}


class DopplerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.img = self.df["pixels"]
        self.lbl = self.df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.img[idx]
        lbl = self.lbl[idx]
        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        return img, lbl


trainset = DopplerDataset(df_train, transform=transform["train"])
valset = DopplerDataset(df_val, transform=transform["test"])
testset = DopplerDataset(df_test, transform=transform["test"])

# WeightedRandomSampler
class_weights = list(1 / df_train["label"].value_counts())
sample_weights = [0] * len(df_train)
for idx, row in df_train.iterrows():
    class_weight = class_weights[row["label"]]
    sample_weights[idx] = class_weight
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

trainLoader = DataLoader(trainset, batch_size=16, sampler=sampler)
valLoader = DataLoader(valset, batch_size=1, shuffle=False)
testLoader = DataLoader(testset, batch_size=1, shuffle=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


def train(epoch, loader, net, model_name):
    net.train()
    totalLoss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Training {model_name}: {epoch+1}/{n_epoch}")
    for X, y in pbar:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = net(X)
        y = y.unsqueeze(dim=1).float()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
        pred = (torch.sigmoid(out) > 0.5).float()
        total += y.size(0)
        correct += pred.eq(y).cpu().sum()
        pbar.set_postfix({"loss": (totalLoss), "acc": (100.0 * correct / total).item()})
    acc = (100.0 * correct / total).item()
    return acc, totalLoss


def evaluate(epoch, loader, net, model_name):
    net.eval()
    totalLoss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Evaluating {model_name}: {epoch+1}/{n_epoch}")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = net(X)
            y = y.unsqueeze(dim=1).float()
            loss = criterion(out, y)
            totalLoss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            total += y.size(0)
            correct += pred.eq(y).cpu().sum()
            pbar.set_postfix(
                {"loss": (totalLoss), "acc": (100.0 * correct / total).item()}
            )
    acc = (100.0 * correct / total).item()
    return acc, totalLoss


model_list = [
    # "inception_v3",
    # "vgg16_bn",
    # "vgg19_bn",
    # "mobilenetv2",
    # "mobilenetv3-small",
    # "mobilenetv3-large",
    # "densenet121",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b1",
]


def get_model(model_name):
    if model_name == "resnet18":
        model = resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif model_name == "resnet50":
        model = resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif model_name == "inception_v3":
        model = inception_v3(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif model_name == "efficientnet_b1":
        model = efficientnet_b1(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif model_name == "vgg16_bn":
        model = vgg16_bn(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)

    elif model_name == "vgg19_bn":
        model = vgg19_bn(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)

    elif model_name == "densenet121":
        model = densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)

    elif model_name == "mobilenetv2":
        model = mobilenet_v2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    elif model_name == "mobilenetv3-small":
        model = mobilenet_v3_small(pretrained=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)

    elif model_name == "mobilenetv3-large":
        model = mobilenet_v3_large(pretrained=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 1)

    else:
        model = None

    return model.to(DEVICE)


train_acc__ = list()
train_loss__ = list()
val_acc__ = list()
val_loss__ = list()
test_acc__ = list()
test_loss__ = list()

n_epoch = 20

for model_name in model_list:
    model = get_model(model_name)
    #     optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=1e-4, nesterov=True, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    train_acc_ = list()
    train_loss_ = list()
    val_acc_ = list()
    val_loss_ = list()
    test_acc_ = list()
    test_loss_ = list()

    for epoch in range(n_epoch):
        train_acc, train_loss = train(epoch, trainLoader, model, model_name)
        val_acc, val_loss = evaluate(epoch, valLoader, model, model_name)
        test_acc, test_loss = evaluate(epoch, testLoader, model, model_name)
        train_acc_.append(train_acc)
        train_loss_.append(train_loss)
        val_acc_.append(val_acc)
        val_loss_.append(val_loss)
        test_acc_.append(test_acc)
        test_loss_.append(test_loss)

    train_acc__.append(train_acc_)
    train_loss__.append(train_loss_)
    val_acc__.append(val_acc_)
    val_loss__.append(val_loss_)
    test_acc__.append(test_acc_)
    test_loss__.append(test_loss_)

    model_path = f"model/xpretrained_20epochs_{model_name}_clean_val.pth"
    torch.save(model, model_path)

color_map = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
for i in range(len(model_list)):
    plt.plot(range(n_epoch), train_loss__[i], color=color_map[i], label=model_list[i])
plt.xticks(range(0, n_epoch, 5))
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training Loss vs Epoch")
plt.legend()

plt.subplot(2, 2, 3)
for i in range(len(model_list)):
    plt.plot(range(n_epoch), val_loss__[i], color=color_map[i], label=model_list[i])
plt.xticks(range(0, n_epoch, 5))
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Validation Loss vs Epoch")
plt.legend()

# plt.subplot(3, 2, 5)
# for i in range(len(model_list)):
#     plt.plot(range(n_epoch), test_loss__[i], color=color_map[i], label=model_list[i])
# plt.xticks(range(0, n_epoch, 5))
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.title("Test Loss vs Epoch")
# plt.legend()

plt.subplot(2, 2, 2)
for i in range(len(model_list)):
    plt.plot(range(n_epoch), train_acc__[i], color=color_map[i], label=model_list[i])
plt.xticks(range(0, n_epoch, 5))
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.legend()

plt.subplot(2, 2, 4)
for i in range(len(model_list)):
    plt.plot(range(n_epoch), val_acc__[i], color=color_map[i], label=model_list[i])
plt.xticks(range(0, n_epoch, 5))
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs Epoch")
plt.legend()

# plt.subplot(3, 2, 6)
# for i in range(len(model_list)):
#     plt.plot(range(n_epoch), test_acc__[i], color=color_map[i], label=model_list[i])
# plt.xticks(range(0, n_epoch, 5))
# plt.xlabel("epochs")
# plt.ylabel("Accuracy")
# plt.title("Test Accuracy vs Epoch")
# plt.legend()

plt.show()


def inference(loader, model_path):
    # DEVICE = "cpu"
    net = torch.load(model_path, map_location=DEVICE)
    net.eval()
    y_ = list()
    pred_ = list()
    prob_ = list()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = net(X)
            y = y.unsqueeze(dim=1).float()
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()
            y_.append(y.squeeze().tolist())
            pred_.append(pred.squeeze().tolist())
            prob_.append((prob * 100.0).squeeze().tolist())
    print(
        classification_report(y_true=y_, y_pred=pred_, target_names=["normal", "fgr"])
    )
    cm = ConfusionMatrixDisplay(
        confusion_matrix(y_true=y_, y_pred=pred_), display_labels=["normal", "fgr"]
    )
    cm.plot()


for model_name in model_list:
    model_test = f"model/xpretrained_20epochs_{model_name}_clean_val.pth"
    print(f"Evaluating {model_name}\n")
    inference(testLoader, model_test)
