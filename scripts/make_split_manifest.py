import pandas as pd

d = pd.concat(
    [pd.read_csv(f"data/splits/{s}.csv").assign(split=s) for s in ("train", "val", "test")]
)
d["path"] = d["path"].str.replace("\\", "/", regex=False)
d.to_csv("data/manifest_split.csv", index=False)
print(len(d), "rows")