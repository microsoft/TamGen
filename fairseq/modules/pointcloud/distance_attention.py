import torch
from torch import nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm

class MultiheadDistanceSelfAttention(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim
        self.num_heads = args.encoder_attention_heads
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim, "embed_dim must be divisible by num_heads"
        self.to_qkv = nn.Linear(self.dim, self.dim * 3)
        self.scaling = self.head_dim ** -0.5
        self.dist_decay = args.dist_decay

        self.softmax = nn.Softmax(dim=2)
        self.dist_embedding = args.dist_embedding
        if self.dist_embedding:
            self.dist_mlp = nn.Sequential(
                nn.Linear(1, self.dim * 4), 
                nn.ReLU(),
                nn.Linear(self.dim * 4, self.dim)
            )
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, pos, mask):
        assert x.shape[0]==pos.shape[0]
        dim = x.shape[-1]
        num_heads, head_dim = self.num_heads, self.head_dim
 
        # compute attention
        # get queries, keys, values
        x_q, x_k, x_v = self.to_qkv(x).chunk(3, dim=-1)
        
        # calculate distance-decayed weight
        dist = torch.linalg.norm((pos.unsqueeze(2) - pos.unsqueeze(1)), dim=-1, keepdim=True)
        if self.dist_embedding:
            dist_embedding = self.dist_mlp(dist) # (b, i, j, dim)

        if self.dist_decay == 0:
            w = 1
        else:
            w = torch.exp(- dist ** 2 / self.dist_decay) # (b, i, j, 1)
        
        # calculate attn weight
        w = w * ((x_k.unsqueeze(2) + (dist_embedding if self.dist_embedding else 0)) * x_q.unsqueeze(1) * self.scaling)
        w = w.contiguous().view(
            x_k.shape[0], x_k.shape[1], x_k.shape[1], num_heads, head_dim
        )  # (b, i, j, num_heads, head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(2) * mask.unsqueeze(1) # (b, i, j)
            w = w.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1), float('-inf'))

        w = self.softmax(w) # (b, i, j, num_heads, head_dim)
        
        x_v = x_v.view(
            x_k.shape[0], 1, x_k.shape[1], num_heads, head_dim
        )  #  (b, i, j, num_heads, head_dim)
        # aggregate
        x = torch.sum(x_v * w, dim=2).contiguous().view(x_k.shape[0], x_k.shape[1], dim)
        x = self.out_proj(x)
        return x

class DistanceTransformerLayer(nn.Module):
    """
    Implements a Point Transformer Layer used in Point Transformer
    models.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim
        self.distance_attention = MultiheadDistanceSelfAttention(args)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.dim)
        self.final_layer_norm = LayerNorm(self.dim)
        self.normalize_before = args.encoder_normalize_before
        self.dropout = args.dropout
        self.self_attn_layer_norm = LayerNorm(self.dim)

    def forward(self, x, pos, mask):
        assert x.shape[0]==pos.shape[0]

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.distance_attention(x, pos, mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x
    
    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x 

