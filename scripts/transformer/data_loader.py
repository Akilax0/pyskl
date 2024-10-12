from pathlib import Path

from torchvision.transforms import Resize, Compose, ToTensor

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define thetrain_transform using Compose
train_transform = Compose([
    Resize((224,224)),
    ToTensor()
])

## Define the test_transform 
test_transform = Compose([
    Resize((224,224)),
    ToTensor()
])

BATCH_SIZE = 256

# Data Directory
data_dir = Path("./data/pizza_steak_sushi")

# Create the training dataset using ImageFolder
training_dataset = ImageFolder(root=data_dir / "train", transform=train_transform)

# Create the test dataset using ImageFolder
test_dataset = ImageFolder(root=data_dir / "test", transform=test_transform)

# Create the training dataloader using DataLoader
training_dataloader = DataLoader(
    dataset=training_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=4,
    drop_last=True
)

# Create the test dataloader using DataLoader
test_dataloader = DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=4,
    drop_last=True
)

