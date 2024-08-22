import torch
from torch import nn
from torch_cluster import fps
from einops import repeat

from  .point_transformer_layer import PointTransformerLayer, knn_query

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=4, nsample=16, keep_points=False):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.ratio = 1/stride
        self.relu = nn.ReLU(inplace=True)
        self.keep_points = keep_points
    
    def forward(self, x, pos, batch):
        assert x.shape[0] == pos.shape[0] == batch.shape[0]
        if self.stride != 1:
            # farthest point sampling
            if not self.keep_points:
                idx = fps(pos, batch, ratio=self.ratio, random_start=True)
            else:
                lengths = torch.bincount(batch)
                keep_points_mask = torch.ceil(lengths*self.ratio).lt(self.nsample)
                if torch.sum(keep_points_mask) > 0:
                    # assuming batch is sorted
                    batch_idx = torch.masked_select(torch.arange(len(lengths)).to(keep_points_mask.device), keep_points_mask)
                    ratio = x.new(len(lengths)).fill_(self.ratio)
                    for i in batch_idx:
                        ratio[i] = self.nsample/lengths[i]
                    idx = fps(pos, batch, ratio=ratio, random_start=True)
                else:
                    idx = fps(pos, batch, ratio=self.ratio, random_start=True)
                
            # grouping
            new_pos = pos[idx, :] # (m, 3)
            new_batch = batch[idx] # (m)
            idx, mask, nsample = knn_query(pos, batch, self.nsample, new_pos, new_batch)
            
            x = torch.cat((pos[idx].transpose(1,2).transpose(0,1).masked_fill_(mask, 0).transpose(0,1).transpose(1,2),\
                x[idx].transpose(1,2).transpose(0,1).masked_fill_(mask, 0).transpose(0,1).transpose(1,2)), -1)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous())) # (m, c, nsample)
            x = x.masked_fill(mask.unsqueeze(1), float('-inf'))
            x = torch.max(x, dim=-1)[0]
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
            new_pos = pos
            new_batch = batch
        return x, new_pos, new_batch
    

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def interpolation(self, pos, new_pos, feat, batch1, batch2, k=3):
        idx, mask, k = knn_query(pos, batch1, k, new_pos, batch2)
        dist = torch.sqrt(torch.sum(torch.square(repeat(new_pos, 'n d -> n k d', k=k) - pos[idx]), dim=-1)) # (n, k)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        dist_recip = dist_recip.masked_fill(mask, 0)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        new_feat = torch.cuda.FloatTensor(new_pos.shape[0], feat.shape[1]).zero_()
        for i in range(k):
            new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
        return new_feat

    def forward(self, x1, pos1, batch1, x2=None, pos2=None, batch2=None):
        assert x1.shape[0] == pos1.shape[0] == batch1.shape[0]
        if x2 is None or pos2 is None or batch2 is None:
            x_tmp = []
            lengths = torch.bincount(batch1)
            offset = 0
            for l in lengths:
                x_b = x1[offset:offset+l]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / l).repeat(l, 1)), 1)
                x_tmp.append(x_b)
                offset += l
            x = self.linear1(torch.cat(x_tmp, 0))
        else:
            assert x2.shape[0] == pos2.shape[0] == batch2.shape[0]
            x = self.linear1(x1) + self.interpolation(pos2, pos1, self.linear2(x2), batch2, batch1)
        return x, pos1, batch1
            
        
class PointTransformerBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(dim=planes, pos_mlp_hidden_dim=planes * 4, nsample=nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pos, batch):
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer(x, pos, batch)))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return x, pos, batch
    