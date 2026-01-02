import os
import requests

url = "https://drive.google.com/uc?id=1c3JqLZhyxpbp1_p6bZ4dTw3Hj9x2z0Yl"
output_dir = "data/attributes/pa100k"
output_file = os.path.join(output_dir, "pa100k.zip")

os.makedirs(output_dir, exist_ok=True)

print("Starting PA-100K download...")

response = requests.get(url, stream=True)
response.raise_for_status()

with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("✅ Download finished:", output_file)
print("File size (MB):", os.path.getsize(output_file) / 1e6)
