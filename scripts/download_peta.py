import os
import requests

url = "https://www.dropbox.com/scl/fi/35o74ndao1aofxodeytmk/PETA.zip?dl=1"
output_dir = "data/attributes"
output_file = os.path.join(output_dir, "PETA.zip")

os.makedirs(output_dir, exist_ok=True)

print("Downloading PETA dataset...")

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

print("✅ Download complete")
print("File size (MB):", os.path.getsize(output_file) / 1e6)
