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

        self.scaling_factor = nn.Parameter(torch.tensor(initial_scaling))
        
        # Learnable node embedding matrix Ψ (N x de)
        self.node_embeddings = nn.Parameter(torch.FloatTensor(node_num, embed_dim))

        self.psi = torch.nn.Parameter(torch.tensor(1.0))
        
        # Learnable filter weights WFilter (N x (K+1) x L)
        self.WFilter = nn.Parameter(torch.FloatTensor(node_num, cheb_k + 1, feature_dim))
        
        # Learnable bias bFilter (N)
        self.bFilter = nn.Parameter(torch.FloatTensor(node_num))
        
        nn.init.kaiming_uniform_(self.node_embeddings, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.WFilter, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bFilter, 0.01) 

    def forward(self, x):
        dist_matrix = torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))  # N x N
        pairwise_distance = torch.diag(dist_matrix)[:, None] + torch.diag(dist_matrix)[None, :] - 2 * dist_matrix
        D = torch.exp(-self.scaling_factor * pairwise_distance)
        adjacency_matrix = F.softmax(self.psi*D, dim=1)  # Ag

        supports = [torch.eye(self.node_num).to(adjacency_matrix.device), adjacency_matrix]
        for k in range(2, self.cheb_k + 1):
            supports.append(2 * torch.matmul(adjacency_matrix, supports[-1]) - supports[-2])

        # Stack supports as (K+1) x N x N
        supports = torch.stack(supports, dim=0)  # (K+1) x N x N

        output = torch.zeros(x.shape[0], self.node_num).to(x.device)
        
        for k in range(self.cheb_k + 1):
            # Multiply supports[k] with self.WFilter[:, k, :] (filter weights)
            support_filter_product = torch.matmul(supports[k], self.WFilter[:, k, :])  # N x L
            
            # Now, multiply this product with the input features x (B x N x L)
            output += torch.einsum('bnl,ln->bl', x, support_filter_product)

        output = output + self.bFilter  

        return output



