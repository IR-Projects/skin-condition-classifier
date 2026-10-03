"""
Downloads just the 'warts' subset from the Fitzpatrick17k dataset.

BEFORE RUNNING:
- fitzpatrick17k.csv must be in the same folder as this script
  (download it from github.com/mattgroh/fitzpatrick17k)
- pip install pandas requests
"""

import os
import pandas as pd
import requests

# --- Step 1: Load the CSV and filter it ---
# pandas reads the whole CSV into a table (a "DataFrame").
df = pd.read_csv("fitzpatrick17k.csv")

# This keeps only the rows where the label column says "warts".
# == is a comparison (is it equal to?), not = (which would assign a value).
warts_df = df[df["label"] == "warts"]

print(f"Found {len(warts_df)} warts entries in the CSV")

# --- Step 2: Make sure the output folder exists ---
outdir = "data/raw/fitzpatrick17k_warts"
os.makedirs(outdir, exist_ok=True)  # exist_ok=True: don't error if it's already there

# --- Step 3: Loop through each row and download the image ---
success_count = 0
fail_count = 0

for index, row in warts_df.iterrows():
    url = row["url"]
    # Build a filename using the row's index so every file gets a unique name.
    filepath = os.path.join(outdir, f"warts_{index}.jpg")

    try:
        # timeout=10: give up on a single image after 10 seconds instead of
        # hanging forever if one URL is dead. This is what makes it
        # "handle failures gracefully" instead of crashing the whole run.
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises an error if the download failed (e.g. 404)

        with open(filepath, "wb") as f:  # "wb" = write bytes (images aren't text)
            f.write(response.content)

        success_count += 1
    except Exception as e:
        # Instead of stopping the whole script on one bad URL, we log it
        # and move on to the next image.
        print(f"Failed: {url} ({e})")
        fail_count += 1

print(f"\nDone. {success_count} downloaded, {fail_count} failed.")
