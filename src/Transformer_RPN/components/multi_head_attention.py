import torch
import torch.nn as nn
from Transformer_RPN.components.self_attention import SelfAttention
class MultiHeadSelfAttention(nn.Module):
  def __init__(self, embedding_dim = 768, num_heads = 12):
    super().__init__()
    self.embeddding_dim = embedding_dim
    self.num_heads = num_heads
    assert embedding_dim % num_heads == 0  
    self.key_dim = embedding_dim // num_heads
    self.attention_list = [SelfAttention(embedding_dim, self.key_dim) for _ in range(num_heads)]
    self.multihead_attention = nn.ModuleList(self.attention_list)
    self.W = nn.Parameter(torch.randn(num_heads * self.key_dim, embedding_dim))
  def forward(self,x):
    attention_scores = [ attention(x) for attention in self.multihead_attention]
    Z = torch.cat(attention_scores, -1)
    attention_score = torch.matmul(Z, self.W)
    return attention_score