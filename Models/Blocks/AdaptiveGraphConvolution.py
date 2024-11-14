import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConvolutionBlock(nn.Module):
    def __init__(self, node_num, embed_dim, cheb_k, feature_dim, initial_scaling=1.0):
        super(AdaptiveGraphConvolutionBlock, self).__init__()
        self.node_num = node_num  # N
        self.embed_dim = embed_dim  # de
        self.cheb_k = cheb_k  # k
        self.feature_dim = feature_dim  # L

        # Learnable scaling factor ψ
        self.scaling_factor = nn.Parameter(torch.tensor(initial_scaling))
        
        # Learnable node embedding matrix Ψ (N x de)
        self.node_embeddings = nn.Parameter(torch.FloatTensor(node_num, embed_dim))
        
        self.Fw = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k + 1, feature_dim))
        self.fb = nn.Parameter(torch.FloatTensor(embed_dim))

        nn.init.kaiming_uniform_(self.node_embeddings, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.Fw, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fb, 0.01)

    def forward(self, x):
        dist_matrix = torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))  # N x N
        pairwise_distance = torch.diag(dist_matrix)[:, None] + torch.diag(dist_matrix)[None, :] - 2 * dist_matrix
        D = torch.exp(-self.scaling_factor * pairwise_distance)
        
        adjacency_matrix = F.softmax(D, dim=1)

        supports = [torch.eye(self.node_num).to(adjacency_matrix.device), adjacency_matrix]
        for k in range(2, self.cheb_k + 1):
            supports.append(2 * torch.matmul(adjacency_matrix, supports[-1]) - supports[-2])

        # Stack supports as (K+1) x N x N
        supports = torch.stack(supports, dim=0)  # (K+1) x N x N

        WFilter = torch.einsum('ij,jkl->ikl', self.node_embeddings, self.Fw)  # N x (K+1) x L
        bFilter = torch.matmul(self.node_embeddings, self.fb)  # N

        output = torch.zeros(x.shape[0], self.node_num).to(x.device)
        
        for k in range(self.cheb_k + 1):
            support_filter_product = torch.matmul(supports[k], WFilter[:, k, :])  # N x L
            
            output += torch.einsum('bnl,nl->bn', x, support_filter_product)
        
        output += bFilter.unsqueeze(0)  
        
        return output


