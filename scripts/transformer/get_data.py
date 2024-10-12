import requests
from pathlib import Path
import os
from zipfile import ZipFile

url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

# Get request to download file
response = requests.get(url)

# Define the path to the data directory
data_path = Path("data")

# Ddedfin the path to image directory
image_path = data_path / "pizza_steak_sushi"

# Check if the image dir already exist
if image_path.is_dir():
    print(f"{image_path} directory already exists")
else:
    print(f"Did not find {image_path} directory, creatiing ...")
    image_path.mkdir(parents=True,exist_ok=True)
    

with open(data_path/"pizza_steak_sushi.zip", "wb") as f:
    f.write(response.content)

# Extract the zip file to the image directory
with ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zipref:
    zipref.extractall(image_path)
    