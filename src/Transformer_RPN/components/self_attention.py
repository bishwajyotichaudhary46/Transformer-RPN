import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
  def __init__(self,embedding_dim, key_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.key_dim = key_dim
    self.W = nn.Parameter(torch.randn(embedding_dim, 3 * key_dim))

  def forward(self,x):
    key_dim = self.key_dim
    qkv = torch.matmul(x,self.W)
    q = qkv[:,:,:key_dim]
    k = qkv[:,:,key_dim:key_dim*2 ]
    v = qkv[:,:,key_dim*2:]
    k_T = torch.transpose(k,-2,-1)
    dot_products = torch.matmul(q,k_T)
    scaled_dot_products = dot_products / np.sqrt(key_dim)
    attention_weights = F.softmax(scaled_dot_products,dim=1)
    weighted_values = torch.matmul(attention_weights,v)
    return weighted_values