import torch
from torch import nn
# from data_loader import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ..builder import BACKBONES

# PATCH_SIZE = 4
# IMAGE_WIDTH = 32
# IMAGE_HEIGHT = IMAGE_WIDTH
# IMAGE_CHANNELS = 3
# EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE**2
# NUM_OF_PATCHES = int((IMAGE_WIDTH*IMAGE_HEIGHT)/PATCH_SIZE**2)

class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,num_joints):
        super().__init__()
        # pe = torch.zeros(num_joints,embedding_dim)
        # print("pe: ",pe.shape)
        # joint_positions = torch.arange(0, num_joints, dtype=torch.float).unsqueeze(1)
        # print("joint positions : ",joint_positions.shape)
        
        # div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        # pe[:, 0::2] = torch.sin(joint_positions * div_term)
        # pe[:, 1::2] = torch.cos(joint_positions * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)

        self.position_embeddings = nn.Parameter(torch.rand((1,num_joints, embedding_dim), requires_grad=True))

    def forward(self, x):
        # print("pe size: ",self.position_embeddings.shape)
        return x + self.position_embeddings
        
@BACKBONES.register_module()
class VisionTransformer(nn.Module):
    def __init__(self,
        image_size = 56, 
        patch_size = 16, 
        embedding_dim = 256,
        heatmap_size = (48,56,56),
        num_joints = 17,
        num_heads = 4,
        num_transformer_layers = 4, 
        mlp_dropout = 0.0,
        attn_dropout = 0.0,
        mlp_size = 48, 
        num_classes = 100,
        in_channels = 3,
        **kwargs
):
        # print("AT INIT")
        # super().__init__(image_size = image_size,patch_size = patch_size, embedding_dim = embedding_dim, num_heads = num_heads, num_transformer_layers = num_transformer_layers, 
        # mlp_dropout = mlp_dropout, attn_dropout = attn_dropout, mlp_size = mlp_size, num_classes = num_classes,
        # in_channels = in_channels, **kwargs)
        super().__init__() 
        
        
        self.heatmap_size = heatmap_size 
        self.embedding_dim = embedding_dim
        self.num_joins = num_joints
        
        self.flatten_size = self.heatmap_size[0]*self.heatmap_size[1]*self.heatmap_size[2]
        # print("fltaten, embedded :", self.flatten_size, embedding_dim)
        # self.embedding = nn.Linear(self.flatten_size, embedding_dim)
        # self.embedding = nn.LazyLinear(embedding_dim)
        self.embedding = None 

        # positional encondings
        self.positional_encoding = PositionalEncoding(embedding_dim, num_joints)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, activation="gelu",batch_first=True)
         
        layernorm = nn.LayerNorm(normalized_shape = embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers, norm=layernorm)

        # Mapping attention to a single score
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self,x):
        
        batch_size, num_joints, temporal_dim, height, width = x.shape
        
        original_heatmaps = x
        
        device = x.device

        # Flatten each heatmap at 48*56*56 
        x = x.view(batch_size, num_joints, -1)
        # print("x flatten: ",x.shape)
        if self.embedding is None:
            flatten_size = x.size(-1)
            self.embedding = nn.Linear(flatten_size, self.embedding_dim).to(device) 
            
        # print("x device:",x.get_device())
        x = self.embedding(x)
        # print("x embedding: ",x.shape)
        x = self.positional_encoding(x)
        # print("x after pos embed: ",x.shape)

        x = self.transformer_encoder(x)
        # print("x encoder output: ",x.shape)
        
        x = self.attention_fc(x)
        # print("x attention vals: ",x.shape)
        

        joint_attention_scores = x.squeeze(-1)  # (batch_size, num_joints)
        # print("joint attention scores: ",joint_attention_scores.shape)
        
        # Softmax across joints to get attention weights
        attention_weights = F.softmax(joint_attention_scores, dim=-1)  # (batch_size, num_joints)
        # print(" attention weights: ",attention_weights.shape)
        
        # Reshape weights for broadcasting over heatmaps
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_joints, 1, 1, 1)
        
        # Apply attention weights to the original heatmaps
        # reweighted_heatmaps = original_heatmaps + original_heatmaps * attention_weights  # (batch_size, num_joints, temporal, H, W) 

        reweighted_heatmaps = original_heatmaps * attention_weights  # (batch_size, num_joints, temporal, H, W) 
        # print(" reweighted weights: ",reweighted_heatmaps.shape)
        # print(" attention weights: ",attention_weights.shape)
        # print("original heatmaps: ",original_heatmaps.shape)
        # print(" attention weights: ",reweighted_heatmaps.shape)
        # print(" attention weights: ",attention_weights)
        
        return reweighted_heatmaps