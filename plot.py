# plot results

import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

path = Path("result/")
data = [str(i) for i in list(path.glob("*.csv"))]

totalData = []
result = {"model": [], "train_acc": [], "val_acc": [], "test_acc": []}

for i in data:
    model = "-".join(i.split("\\")[-1].split("-")[:2])
    df_temp = pd.read_csv(i)
    df_temp["model"] = model
    totalData.append(df_temp)
    best_score_idx = df_temp[["val_acc"]].idxmax()
    result["model"].append(model)
    result["train_acc"].append(df_temp["train_acc"][best_score_idx].item())
    result["val_acc"].append(df_temp["val_acc"][best_score_idx].item())
    result["test_acc"].append(df_temp["test_acc"][best_score_idx].item())

df = pd.concat(totalData, ignore_index=True)
df.rename(columns={"Unnamed: 0": "epoch"}, inplace=True)
df["epoch"] = [i + 1 for i in df["epoch"]]

for i in df.columns[1:-1]:
    fig = px.line(df, x="epoch", y=i, color="model", markers=True, title=i)
    fig.show()

fig = go.Figure(
    data=[
        go.Bar(
            name="test_acc", y=result["model"], x=result["test_acc"], orientation="h"
        ),
        go.Bar(name="val_acc", y=result["model"], x=result["val_acc"], orientation="h"),
        go.Bar(
            name="train_acc", y=result["model"], x=result["train_acc"], orientation="h"
        ),
    ]
)

fig.update_layout(
    title="best val results",
    barmode="stack",
    yaxis={"categoryorder": "total ascending"},
)

fig.show()
