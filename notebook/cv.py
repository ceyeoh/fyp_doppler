import numpy as np
import torch
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pickle
from model import get_model
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

path = Path("data/preprocess")
data = [str(i) for i in list(path.glob("*.jpg"))]

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

df_cv = pd.concat([df_duplicated, df_unique], ignore_index=True)

scv = StratifiedKFold(n_splits=8, shuffle=True, random_state=SEED)
x_cv = df_cv.drop(columns=["label"])
y_cv = df_cv["label"]
cv_set = {
    "train": [],
    "val": [],
}
for idx, (train_index, val_index) in enumerate(scv.split(x_cv, y_cv)):
    # print(f"Split {idx}")
    # print(f"Total train: {len(train_index)}")
    # print(f"Total val: {len(val_index)}")
    cv_set["train"].append(df_cv.loc[train_index])
    cv_set["val"].append(df_cv.loc[val_index])
    # print("Val label:", cv_set["val"][idx]["label"].tolist())
    # print("\n")

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


def train(epoch, loader, net, model_name, split):
    net.train()
    totalLoss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Training split {split} {model_name}: {epoch+1}/{n_epoch}")
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


def evaluate_val(epoch, loader, net, model_name, split):
    net.eval()
    totalLoss = 0
    correct = 0
    total = 2
    pbar = tqdm(loader, desc=f"Evaluating split {split} {model_name}: {epoch+1}/{n_epoch}")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = net(X)
            y = y.unsqueeze(dim=1).float()
            loss = criterion(out, y)
            totalLoss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            # total += y.size(0)
            weight = 1/val_counts[1] if y.item()==1 else 1/val_counts[0]
            correct += pred.eq(y).cpu().sum()*weight
            pbar.set_postfix(
                {"loss": (totalLoss), "acc": (100.0 * correct / total).item()}
            )
    acc = (100.0 * correct / total).item()
    return acc, totalLoss


def evaluate_test(epoch, loader, net, model_name, split):
    net.eval()
    totalLoss = 0
    correct = 0
    total = 2
    pbar = tqdm(loader, desc=f"Evaluating split {split} {model_name}: {epoch+1}/{n_epoch}")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = net(X)
            y = y.unsqueeze(dim=1).float()
            loss = criterion(out, y)
            totalLoss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            # total += y.size(0)
            weight = 1/test_counts[1] if y.item()==1 else 1/test_counts[0]
            correct += pred.eq(y).cpu().sum()*weight
            pbar.set_postfix(
                {"loss": (totalLoss), "acc": (100.0 * correct / total).item()}
            )
    acc = (100.0 * correct / total).item()
    return acc, totalLoss

# def evaluate_test(epoch, loader, net, model_name, split):
#     net.eval()
#     totalLoss = 0
#     correct = 0
#     total = 0
#     pbar = tqdm(loader, desc=f"Evaluating split {split} {model_name}: {epoch+1}/{n_epoch}")
#     with torch.no_grad():
#         for X, y in pbar:
#             X, y = X.to(DEVICE), y.to(DEVICE)
#             out = net(X)
#             y = y.unsqueeze(dim=1).float()
#             loss = criterion(out, y)
#             totalLoss += loss.item()
#             pred = (torch.sigmoid(out) > 0.5).float()
#             total += y.size(0)
#             correct += pred.eq(y).cpu().sum()
#             pbar.set_postfix(
#                 {"loss": (totalLoss), "acc": (100.0 * correct / total).item()}
#             )
#     acc = (100.0 * correct / total).item()
#     return acc, totalLoss


testset = DopplerDataset(df_test, transform=transform["test"])
testLoader = DataLoader(testset, batch_size=1, shuffle=False)
test_counts = df_test["label"].value_counts()


results = dict(
    (
        splits,
        dict(
            (metrics, list())
            for metrics in [
                "train_acc",
                "train_loss",
                "val_acc",
                "val_loss",
                "test_acc",
                "test_loss",
            ]
        ),
    )
    for splits in range(8)
)
for split in range(len(cv_set["val"])):
    df_train = cv_set["train"][split].reset_index(drop=True)
    df_val = cv_set["val"][split].reset_index(drop=True)
    trainset = DopplerDataset(df_train, transform=transform["train"])
    valset = DopplerDataset(df_val, transform=transform["test"])
    model_name = "mobilenet_v3_large-pretrained-adam"
    model = get_model(model_name).to(DEVICE)
    lr = 1e-4
    n_epoch = 25
    bs = 16
    # filename = f"{model_name}-{lr}lr-{bs}bs-{n_epoch}e-split{split}-loss-weighted"
    filename = f"{model_name}-{lr}lr-{bs}bs-{n_epoch}e-split{split}-loss-normal"
    train_counts = df_train["label"].value_counts()
    val_counts = df_val["label"].value_counts()
    class_weights = list(1 / train_counts)
    sample_weights = [0] * len(df_train)
    for idx, row in df_train.iterrows():
        class_weight = class_weights[row["label"]]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    trainLoader = DataLoader(trainset, batch_size=bs, sampler=sampler)
    valLoader = DataLoader(valset, batch_size=1, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_counts[1]/train_counts[0]))
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 60.0
    for epoch in range(n_epoch):
        train_acc, train_loss = train(epoch, trainLoader, model, model_name, split)
        val_acc, val_loss = evaluate_val(epoch, valLoader, model, model_name, split)
        test_acc, test_loss = evaluate_test(epoch, testLoader, model, model_name, split)
        results[split]["train_acc"].append(train_acc)
        results[split]["train_loss"].append(train_loss)
        results[split]["val_acc"].append(val_acc)
        results[split]["val_loss"].append(val_loss)
        results[split]["test_acc"].append(test_acc)
        results[split]["test_loss"].append(test_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Best val {best_acc}, saving...")
            # torch.save(model, f"../model/{filename}-{best_acc}val-{epoch}e.pth")

a_file = open(f"result_cv/{filename}.pkl", "wb")
pickle.dump(results, a_file)
a_file.close()





