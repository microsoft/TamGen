import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn, knn_graph
from einops import repeat, rearrange
from fairseq.modules import LayerNorm
import time

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def knn_query(pos, batch, k, new_pos=None, new_batch=None):
    if new_pos is not None:
        lengths, tmp_idx = knn(pos, new_pos, k=k, batch_x=batch, batch_y=new_batch)
    else:
        tmp_idx, lengths = knn_graph(pos, k=k, batch=batch, loop=True)
    lengths = torch.bincount(lengths)
    if lengths.min() == lengths.max():
        k = lengths[0]
        idx = tmp_idx.contiguous().view(-1, k)
        mask = idx.eq(-1)
    else: 
        k = lengths.max() if k>lengths.max() else k
        offset = 0
        idx = tmp_idx.new(len(lengths), k).fill_(-1)
        for i, l in enumerate(lengths):
            if k > l:
                idx[i][:l] = tmp_idx[offset:offset+l]
            else:
                idx[i] = tmp_idx[offset:offset+k]
            offset += l
        mask = idx.eq(-1)
    return idx, mask, k


class MultiheadLocalPosAttention(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim if args.no_cat else args.encoder_embed_dim + args.encoder_coord_dim
        self.nsample = args.nsample
        self.num_heads = args.encoder_attention_heads
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim, "embed_dim must be divisible by num_heads"
        self.to_qkv = Linear(self.dim, self.dim * 3)
        self.use_distance = args.use_distance
        self.scale = args.pos_mlp_scale
        self.value_without_relpos = args.value_without_relpos
        self.attn_weight_type = args.attn_weight_type
        
        assert self.attn_weight_type in ['normal', 'linear', 'origin_mlp']

        pos_mlp_hidden_dim = self.dim // 2
        self.pos_mlp = nn.Sequential(
            Linear(1 if self.use_distance else 3, pos_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            Linear(pos_mlp_hidden_dim, self.dim)
        )

        if self.attn_weight_type == 'origin_mlp' or args.attn_weight_mlp:
            self.attn_mlp = nn.Sequential(
                Linear(self.head_dim, self.head_dim * 4),
                nn.ReLU(inplace=True),
                Linear(self.head_dim * 4, self.head_dim),
            )
        else:
            self.attn_mlp = None
        if self.attn_weight_type == 'linear':
            self.w1 = nn.Parameter(torch.Tensor(1))
            self.w2 = nn.Parameter(torch.Tensor(1))
            self.w3 = nn.Parameter(torch.Tensor(1))
        self.softmax = nn.Softmax(dim=1)
        self.out_proj = Linear(self.dim, self.dim)


    def forward(self, x, pos, attn_index, mask, nsample):
        assert x.shape[0]==pos.shape[0]
        dim = x.shape[-1]
        num_heads, head_dim = self.num_heads, self.head_dim
 
        # compute attention
        # get queries, keys, values
        x_q, x_k, x_v = self.to_qkv(x).chunk(3, dim=-1)
        
        # (n, nsample, dim)
        x_k = x_k[attn_index]
        x_v = x_v[attn_index]
        
        # calculate relative positional embeddings
        rel_pos = pos[attn_index] - pos.unsqueeze(1) # (n, nsample, 3)
        if self.use_distance:
            rel_pos = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        # pos_mlp
        # add scale
        rel_pos = self.scale * self.pos_mlp(rel_pos) # (n, nsample, dim)

        # calculate attn weight
        # attn_mlp: 改成其他的，如：w1*k+w2*q+w3*rel_pos
        if self.attn_weight_type == 'normal':
            w = x_k * x_q.unsqueeze(1) + rel_pos
        elif self.attn_weight_type == 'linear':
            w = self.w1 * x_k + self.w2 * x_q.unsqueeze(1) + self.w3 * rel_pos
        else:
            w = x_k - x_q.unsqueeze(1) + rel_pos
        w = w.contiguous().view(
            x_k.shape[0], nsample, num_heads, head_dim
        )  # (n, nsample, num_heads, head_dim)
        if self.attn_mlp is not None:
            w = self.attn_mlp(w)
        w = w.transpose(0,2).transpose(1,3).masked_fill_(
            mask, float('-inf')
        ).transpose(1,3).transpose(0,2)
        w = self.softmax(w) # (n, nsample, num_heads, head_dim)
        
        # add relative positional embeddings to value
        # 去掉 rel_pos
        if not self.value_without_relpos:
            x_v = (x_v + rel_pos).contiguous().view(
                x_k.shape[0], nsample, num_heads, head_dim
            )  # (n, nsample, num_heads, head_dim)
        else:
            x_v = x_v.view(
                x_k.shape[0], nsample, num_heads, head_dim
            )  # (n, nsample, num_heads, head_dim)
        # aggregate
        x = torch.sum(x_v * w, dim=1).contiguous().view(-1, dim)
        x = self.out_proj(x)
        return x

class MultiheadGlobalPosAttention(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim if args.no_cat else args.encoder_embed_dim + args.encoder_coord_dim
        self.num_heads = args.encoder_attention_heads
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim, "embed_dim must be divisible by num_heads"
        self.to_qkv = Linear(self.dim, self.dim * 3)
        self.use_distance = args.use_distance
        self.scale = args.pos_mlp_scale
        self.value_without_relpos = args.value_without_relpos
        self.attn_weight_type = args.attn_weight_type
        
        assert self.attn_weight_type in ['normal', 'linear', 'origin_mlp']

        pos_mlp_hidden_dim = self.dim // 2
        self.pos_mlp = nn.Sequential(
            Linear(1 if self.use_distance else 3, pos_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            Linear(pos_mlp_hidden_dim, self.dim)
        )

        if self.attn_weight_type == 'origin_mlp' or args.attn_weight_mlp:
            self.attn_mlp = nn.Sequential(
                Linear(self.head_dim, self.head_dim * 4),
                nn.ReLU(inplace=True),
                Linear(self.head_dim * 4, self.head_dim),
            )
        else:
            self.attn_mlp = None
        if self.attn_weight_type == 'linear':
            self.w1 = nn.Parameter(torch.Tensor(1))
            self.w2 = nn.Parameter(torch.Tensor(1))
            self.w3 = nn.Parameter(torch.Tensor(1))
        self.softmax = nn.Softmax(dim=2)
        self.out_proj = Linear(self.dim, self.dim)

    def forward(self, x, pos, mask):
        assert x.shape[0]==pos.shape[0]
        dim = x.shape[-1]
        num_heads, head_dim = self.num_heads, self.head_dim
 
        # compute attention
        # get queries, keys, values
        x_q, x_k, x_v = self.to_qkv(x).chunk(3, dim=-1)
        
        # calculate relative positional embeddings
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1) # (b, i, j, 3)
        if self.use_distance:
            rel_pos = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        # pos_mlp
        # add scale
        rel_pos = self.scale * self.pos_mlp(rel_pos) # (b, i, j, dim)

        # calculate attn weight
        # attn_mlp: 改成其他的，如：w1*k+w2*q+w3*rel_pos
        if self.attn_weight_type == 'normal':
            w = x_k.unsqueeze(2) * x_q.unsqueeze(1) + rel_pos
        elif self.attn_weight_type == 'linear':
            w = self.w1 * x_k.unsqueeze(2) + self.w2 * x_q.unsqueeze(1) + self.w3 * rel_pos
        else:
            w = x_k.unsqueeze(2) - x_q.unsqueeze(1) + rel_pos
        w = w.contiguous().view(
            x_k.shape[0], x_k.shape[1], x_k.shape[1], num_heads, head_dim
        )  # (b, i, j, num_heads, head_dim)
        if self.attn_mlp is not None:
            w = self.attn_mlp(w)
        
        mask = mask.unsqueeze(2) * mask.unsqueeze(1) # (b, i, j)
        w = w.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1), float('-inf'))
        w = self.softmax(w) # (b, i, j, num_heads, head_dim)
        
        # add relative positional embeddings to value
        # 去掉 rel_pos
        if not self.value_without_relpos:
            x_v = (x_v.unsqueeze(1) + rel_pos).contiguous().view(
                x_k.shape[0], x_k.shape[1], x_k.shape[1], num_heads, head_dim
            )  # (b, i, j, num_heads, head_dim)
        else:
            x_v = x_v.view(
                x_k.shape[0], x_k.shape[1], x_k.shape[1], num_heads, head_dim
            )  #  (b, i, j, num_heads, head_dim)
        # aggregate
        x = torch.sum(x_v * w, dim=2).contiguous().view(x_k.shape[0], x_k.shape[1], dim)
        x = self.out_proj(x)
        return x

    # def forward(self, x, pos, mask):
    #     assert x.shape[0]==pos.shape[0]
    #     n, h = x.shape[1], self.num_heads
        
    #     # get queries, keys, values
    #     q, k, v = self.to_qkv(x).chunk(3, dim = -1)

    #     # split out heads
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

    #     # calculate relative positional embeddings
    #     rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
    #     if self.use_distance:
    #         rel_pos = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    #     rel_pos_emb = self.pos_mlp(rel_pos)

    #     # split out heads for rel pos emb
    #     rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h = h)

    #     # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
    #     qk_rel = q.unsqueeze(3) - k.unsqueeze(2)

    #     # prepare mask
    #     mask = mask.unsqueeze(2) * mask.unsqueeze(1)

    #     # add relative positional embeddings to value
    #     v = v.unsqueeze(2) + rel_pos_emb

    #     # use attention mlp, making sure to add relative positional embedding first
    #     attn_mlp_input = qk_rel + rel_pos_emb
    #     # attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

    #     sim = self.attn_mlp(attn_mlp_input)
    #     # masking
    #     sim.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), float('-inf'))

    #     # attention
    #     attn = sim.softmax(dim = 2)

    #     # aggregate
    #     agg = torch.einsum('b h i j d, b h i j d -> b h i d', attn, v)
    #     agg = rearrange(agg, 'b h n d -> b n (h d)')

    #     # combine heads
    #     return self.out_proj(agg)
    
class PointTransformerLayer(nn.Module):
    """
    Implements a Point Transformer Layer used in Point Transformer
    models.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim if args.no_cat else args.encoder_embed_dim + args.encoder_coord_dim
        self.pos_attention = MultiheadLocalPosAttention(args)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = Linear(self.dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.dim)
        self.final_layer_norm = LayerNorm(self.dim)
        self.normalize_before = args.encoder_normalize_before
        self.dropout = args.dropout
        self.self_attn_layer_norm = LayerNorm(self.dim)

    def forward(self, x, pos, attn_index, mask, nsample):
        assert x.shape[0]==pos.shape[0]

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.pos_attention(x, pos, attn_index, mask, nsample)
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
        
class PointTransformerLayerGlobal(nn.Module):
    """
    Implements a Point Transformer Layer used in Point Transformer
    models.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.encoder_embed_dim if args.no_cat else args.encoder_embed_dim + args.encoder_coord_dim
        self.pos_attention = MultiheadGlobalPosAttention(args)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = Linear(self.dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.dim)
        self.final_layer_norm = LayerNorm(self.dim)
        self.normalize_before = args.encoder_normalize_before
        self.dropout = args.dropout
        self.self_attn_layer_norm = LayerNorm(self.dim)

    def forward(self, x, pos, mask):
        assert x.shape[0]==pos.shape[0]

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.pos_attention(x, pos, mask)
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
