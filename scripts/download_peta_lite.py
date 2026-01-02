import os
import urllib.request


os.makedirs("data/attributes/peta_lite", exist_ok=True)

url = "https://github.com/CVMI-Lab/PETA/releases/download/v1.0/peta_lite.zip"
destination = "data/attributes/peta_lite/peta_lite.zip"

print("Downloading PETA-Lite...")
urllib.request.urlretrieve(url, destination)

print("Downloaded to:", destination)
