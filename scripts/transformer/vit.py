import torch
from torch import nn
# from data_loader import *
import matplotlib.pyplot as plt

# PATCH_SIZE = 4
# IMAGE_WIDTH = 32
# IMAGE_HEIGHT = IMAGE_WIDTH
# IMAGE_CHANNELS = 3
# EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE**2
# NUM_OF_PATCHES = int((IMAGE_WIDTH*IMAGE_HEIGHT)/PATCH_SIZE**2)

# The image width and height should be divisible by patchy size 
# assert IMAGE_WIDTH % PATCH_SIZE == 0 and IMAGE_HEIGHT % PATCH_SIZE == 0, "IMAGE width is not divisible by patch size"

# Takes image and throws patch embeddings : image embedding + class token + position embedding
class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim,batch_size, num_of_patches):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, \
            kernel_size=patch_size, stride=patch_size)
        self.flatten_layer = nn.Flatten(start_dim = 1, end_dim=2)
        self.class_token_embeddings = nn.Parameter(torch.rand((1, 1, embedding_dim), requires_grad = True))
        self.position_embeddings = nn.Parameter(torch.rand((1,num_of_patches + 1, embedding_dim), requires_grad=True))
        
        
    def forward(self,x):
        
        
        # print("x: ",x.shape)
        conv =  self.conv_layer(x).permute((0,2,3,1))
        # print("conv: ",conv.shape)
        flat =  self.flatten_layer(conv)
        # print("flat: ",flat.shape)
        
        batch_size, _, _ = flat.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.class_token_embeddings.expand(batch_size, -1, -1)
        
        cat = torch.cat((cls_tokens,flat),dim=1)
        # print("cat: ",cat.shape)
        output = cat + self.position_embeddings
        
        # output = torch.cat((self.class_token_embeddings, self.flatten_layer(self.conv_layer(x).permute((0,2,3,1)))), dim=1) \
        #     + self.position_embeddings
            
        return output
        
# Multi-Head Self Attnetion Block 
# Layer Norm : normalize patch embeddings across embedding dimension
# Multi Head: Q,K,V 
# input and output size same for the MSA block - [batch_size, sequence_length, embedding_dimensions]


# MSA block defined acccording to the parameters in the ViT paper
# ViT-Base is implemented here

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, 
                 embedding_dims = 768, # Hidden size D in ViT
                 num_heads = 12, # Heads in the paper
                 attn_dropout = 0.0 # Defaults to zero as no there is no dropout for the block
    ):
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.num_head = num_heads
        self.attn_dropout = attn_dropout
        
        self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
        self.multiheadattention = nn.MultiheadAttention(num_heads = num_heads,
                                                        embed_dim = embedding_dims,
                                                        dropout = attn_dropout,
                                                        batch_first = True,)
        
        
    def forward(self,x):
        x = self.layernorm(x)
        output,_ = self.multiheadattention(query=x, key=x, value=x, need_weights=False)
        
        return output
    
    
# MLP Block Code
class MachineLearningPerceptronBlock(nn.Module):
    def __init__ (self, embedding_dims, mlp_size, mlp_dropout):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.mlp_size = mlp_size
        self.dropout = mlp_dropout
        
        self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dims, out_features = mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features = mlp_size, out_features = embedding_dims),
            nn.Dropout(p=mlp_dropout)
        )
        
    def forward(self, x):
        return self.mlp(self.layernorm(x))
        
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims = 48,
                 mlp_dropout = 0.1,
                 attn_dropout = 0.0,
                 mlp_size = 3072,
                 num_heads = 4,
                 ):
        super().__init__()
        
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims= embedding_dims,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        self.mlp_block = MachineLearningPerceptronBlock(embedding_dims=embedding_dims,
                                                        mlp_size=mlp_size,
                                                        mlp_dropout=mlp_dropout)
        
    def forward(self,x):
        x = self.msa_block(x) + x 
        x = self.mlp_block(x) + x
        return x
    
    
class ViT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.mlp_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.in_channels = config["num_channels"]
        self.patch_size = config["patch_size"]
        self.embedding_dim = config["hidden_size"]
        self.mlp_dropout = config["hidden_dropout_prob"]
        self.attn_dropout = config["attention_probs_dropout_prob"]
        self.num_heads = config["num_attention_heads"]
        self.num_transformer_layers = config["num_hidden_layers"]
        self.num_of_patches = int((self.image_size ** 2) / self.patch_size **2)
        self.batch_size = config["batch_size"]
# NUM_OF_PATCHES = int((IMAGE_WIDTH*IMAGE_HEIGHT)/PATCH_SIZE**2)
        
        self.patch_embedding_layer = PatchEmbeddingLayer(in_channels=self.in_channels,
                                                         patch_size = self.patch_size,
                                                         embedding_dim=self.embedding_dim,
                                                         batch_size=self.batch_size,
                                                         num_of_patches=self.num_of_patches)
        self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dims=self.embedding_dim,
                                                                    mlp_dropout=self.mlp_dropout,
                                                                    attn_dropout=self.attn_dropout,
                                                                    mlp_size=self.mlp_size,
                                                                    num_heads=self.num_heads)
                                                   for _ in range(self.num_transformer_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=self.embedding_dim),
                                        nn.Linear(in_features = self.embedding_dim,
                                                  out_features = self.num_classes))
        
    def forward(self,x):
        return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:,0])