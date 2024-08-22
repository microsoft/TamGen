import torch
from torch import nn
from torch_cluster import knn, knn_graph

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

class PointTransformerLayer(nn.Module):
    """
    Implements a Point Transformer Layer used in Point Transformer
    models.
    [TODO]: Multihead version
    """
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_heads = 4,
        nsample = 16,
    ):
        super().__init__()
        self.nsample = nsample
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        assert self.head_dim * num_heads == dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, pos, batch):
        assert x.shape[0]==pos.shape[0]==batch.shape[0]

        # get queries, keys, values
        x_q, x_k, x_v = self.to_qkv(x).chunk(3, dim = -1)
        
        attn_index, mask, nsample = knn_query(pos, batch, self.nsample)

        x_k = x_k[attn_index] # (n, nsample, dim)
        x_v = x_v[attn_index] # (n, nsample, dim)
        
        # calculate relative positional embeddings
        rel_pos = pos[attn_index] - pos.unsqueeze(1) # (n, nsample, 3)
        
        # pos_mlp
        rel_pos = self.pos_mlp(rel_pos)
        
        # calculate attn weight
        w = (x_k - x_q.unsqueeze(1) + rel_pos)  # (n, nsample, dim)
        w = self.attn_mlp(w)
        w = w.transpose(1,2).transpose(0,1).masked_fill_(mask, float('-inf')).transpose(0,1).transpose(1,2)
        w = self.softmax(w) # (n, nsample, dim) 
        
        # add relative positional embeddings to value
        x_v = x_v + rel_pos

        # aggregate
        agg = torch.sum(x_v * w, dim=1)
        return agg
