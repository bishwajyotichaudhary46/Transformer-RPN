import torch
import torch.nn as nn
from Transformer_RPN.components.encoder import TransformerEncoder

class VisionTransformer(nn.Module):
  
  def __init__(self,patch_size=16,image_size=224,channel_size=3,num_layers=12,embedding_dim=768,num_heads=12,
               hidden_dim=3072,dropout_prob=0.1):
    
    super().__init__()
    self.patch_size = patch_size
    self.channel_size = channel_size
    self.num_layers = num_layers
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.dropout_prob = dropout_prob
    self.dimension = embedding_dim

    self.num_patches = int(image_size**2 / patch_size**2)
    self.W = nn.Parameter(torch.randn(patch_size *patch_size *channel_size, embedding_dim))
    self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embedding_dim))
    self.class_token = nn.Parameter(torch.randn(1, self.dimension))

    transformer_encoder_list = [
        TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob)
        for _ in range(num_layers)
    ]
    self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)

  def forward(self,x):
    P,C = self.patch_size,self.channel_size
    
    #[batch_size,channels,hight,width]
    patches = x.unfold(1,C,C).unfold(2,P,P).unfold(3,P,P)
    patches = patches.contiguous().view(patches.size(0),-1,C*P*P).float()

    patch_embeddings = torch.matmul(patches,self.W)

    batch_size = patch_embeddings.shape[0]
    patch_embeddings = torch.cat((self.class_token.repeat(batch_size,1,1),patch_embeddings),1)

    patch_embeddings = patch_embeddings + self.pos_embedding
    transformer_encoder_output = self.transformer_encoder_layers(patch_embeddings)

    return transformer_encoder_output