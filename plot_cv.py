import pandas as pd
import pickle

from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# path = Path("result_cv/")
# data = [str(i) for i in list(path.glob("*.pkl"))]

# print(data)
a_file = open(
    "result_cv/densenet121-pretrained-adam-0.0001lr-16bs-25e-split7-loss-normal.pkl",
    "rb",
)
model_name = "-".join(a_file.name.split("/")[-1].split("-")[:2])
# print(model_name)
results = pickle.load(a_file)

train_acc = [results[i]["train_acc"] for i in results]
train_loss = [results[i]["train_loss"] for i in results]
val_acc = [results[i]["val_acc"] for i in results]
val_loss = [results[i]["val_loss"] for i in results]
test_acc = [results[i]["test_acc"] for i in results]
test_loss = [results[i]["test_loss"] for i in results]

df = pd.DataFrame(
    columns=[
        "split",
        "train_acc",
        "train_loss",
        "val_acc",
        "val_loss",
        "test_acc",
        "test_loss",
        "epoch",
    ]
)

for split in results:
    for epoch in range(len(train_acc[split])):
        df = df.append(
            {
                "split": split,
                "train_acc": train_acc[split][epoch],
                "train_loss": train_loss[split][epoch],
                "val_acc": val_acc[split][epoch],
                "val_loss": val_loss[split][epoch],
                "test_acc": test_acc[split][epoch],
                "test_loss": test_loss[split][epoch],
                "epoch": epoch + 1,
            },
            ignore_index=True,
        )

for i in df.columns[1:-1]:
    fig = px.line(
        df, x="epoch", y=i, color="split", markers=True, title=model_name + ": " + i
    )
    fig.show()
