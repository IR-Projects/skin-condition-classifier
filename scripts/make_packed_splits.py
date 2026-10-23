import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

zip_path, out_dir = sys.argv[1], Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    m = pd.read_csv(io.BytesIO(zf.read("packed_manifest.csv")))
out_dir.mkdir(parents=True, exist_ok=True)
for s in ("train", "val", "test"):
    sub = m[m["split"] == s].copy()
    sub["path"] = sub["packed_path"]
    sub.drop(columns=["packed_path", "split"]).to_csv(out_dir / f"{s}.csv", index=False)
    print(s, len(sub))