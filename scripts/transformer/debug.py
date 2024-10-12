import torch
from torch import nn
from data_loader import *
import matplotlib.pyplot as plt

PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE**2
NUM_OF_PATCHES = int((IMAGE_WIDTH*IMAGE_HEIGHT)/PATCH_SIZE**2)

# The image width and height should be divisible by patchy size 
assert IMAGE_WIDTH % PATCH_SIZE == 0 and IMAGE_HEIGHT % PATCH_SIZE == 0, "IMAGE width is not divisible by patch size"


# Converting image to patches and creating an embedding vector for each patch size 768

conv_layer = nn.Conv2d(in_channels = IMAGE_CHANNELS, out_channels = EMBEDDING_DIMS, kernel_size = PATCH_SIZE, stride = PATCH_SIZE)


# ===================  Passing an image to check ====================================

print(len(training_dataloader.dataset))
random_images, random_labels = next(iter(training_dataloader))
print(len(random_images), len(random_labels))
random_image = random_images[0]


# Create a new figure

fig = plt.figure(1)

print(random_image.size())
# Random image save 
plt.imshow(random_image.permute((1,2,0)))
print(random_image.permute(1,2,0).size())

# Disable the axis for visualization 
plt.axis(False)

plt.title(training_dataset.classes[random_labels[0]], color="white")

fig.set_facecolor(color="black")

# Saving random image
plt.savefig("random_image.png")

# Reshaping iamge to [1,14,14,768]
# And flaten output to [1, 196, 768]

image_through_conv = conv_layer(random_image.unsqueeze(0))
print(f'Shapeof embeddings through the conv layer -> {list(image_through_conv.shape)} <- [batch_size, num_of_patch_rows, num_path_cols embedding_dims]')

# Permute the dimenstions of image_through_conv to match expected shape
image_through_conv = image_through_conv.permute((0,2,3,1))
# print(image_through_conv.size())

# Crete a flatten layer using nn.Flatten
flatten_layer = nn.Flatten(start_dim = 1, end_dim =2)

# Pass the image through conv through the flatten layer
image_through_conv_and_flatten = flatten_layer(image_through_conv)

#print the shape oif the embedding image
print(f'Shape of embeddings through the flatten layer -> {list(image_through_conv_and_flatten.shape)} <- [batch_size, num_of_patches, embedding_dims]')

# Assign the embedded image to a variable
embedded_image = image_through_conv_and_flatten


################### Class token + Pos embeddings ##################################

class_token_embeddings = nn.Parameter(torch.rand((1,1,EMBEDDING_DIMS), requires_grad = True))
print(f'Shape of clas_token_embeddings --> {list(class_token_embeddings.shape)} <-- [batch_size , 1 ,embedding_dims]')

embedded_image_with_class_token_embeddings = torch.cat((class_token_embeddings,embedded_image),dim=1)
print(f'\n Shape of image embeddings with class_token_embeddings --> {list(embedded_image_with_class_token_embeddings.shape)} <-- [batch_size, num_patches+1, embeddings_dims]')

position_embeddings = nn.Parameter(torch.rand((1, NUM_OF_PATCHES+1, EMBEDDING_DIMS), requires_grad = True))
print(f'\nShape of position_embeddings --> {list(position_embeddings.shape)} <-- [batch_size, num_patches+1, embeddings_dims]')

final_embeddings = embedded_image_with_class_token_embeddings + position_embeddings
print(f'\nShape of final_embeddings --> {list(final_embeddings.shape)} <-- [batch_size, num_patches+1, embeddings_dims]')


# ===================================================================================