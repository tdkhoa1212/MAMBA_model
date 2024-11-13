import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
from Models.Blocks.BidirectionalMamba import BidirectionalMambaBlock
from Models.Blocks.AdaptiveGraphConvolution import AdaptiveGraphConvolutionBlock
# from torch_geometric.typing import torch_cluster

class GRAPH_MAMBA(nn.Module):
    def __init__(self, configs):
        super(GRAPH_MAMBA, self).__init__()
        self.configs = configs
        
        # Initialize the Mamba block: Bidirectional Mamba layer for sequence modeling
        self.mamba_block = BidirectionalMambaBlock(
            pred_len=configs.pred_len, 
            d_model=configs.d_model, 
            d_state=configs.d_state, 
            seq_len=configs.seq_len, 
            num_layers=configs.num_layers,
            d_inner=configs.d_inner,
            hidden_dimention=configs.hidden_dimention
        )

        # Initialize the Adaptive Graph Convolution Block
        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            cheb_k=configs.cheb_k, 
            feature_dim=configs.feature_dim,
        )

        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            feature_dim=configs.feature_dim,
            cheb_k=configs.cheb_k
        )

        # Regularization parameters
        self.l2_lambda = 1e-4
        self.dropout = nn.Dropout(p=0.1)

        # GCN Layer: GCNConv layer to apply graph convolutions
        # self.gcn = GCNConv(configs.feature_dim, 1, improved=True, bias=True)

        # Projection layer: Linear layer for the final prediction
        self.projection = nn.Linear(configs.linear_depth, configs.pred_len, bias=True)

        # Learnable scaling factor for Gaussian kernel in the adaptive graph construction
        self.psi = torch.nn.Parameter(torch.tensor(1.0))  # ψ is initialized as learnable scaler
        self.flatten = nn.Flatten()

        # Node embedding matrix Ψ (N x de), initialized randomly
        self.Psi = torch.nn.Parameter(torch.randn(configs.node_num, configs.embed_dim))  # learnable embedding matrix Ψ

    def forward(self, input_):
        """
        Forward pass through the model:
        1. Apply the Mamba block to process the input sequence.
        2. Apply the GCN layer to capture graph-based dependencies.
        3. Apply the final linear projection to get the prediction.
        """
        x = input_
        device = input_.device
        
        # Pass through the Mamba blocks
        for i in range(self.configs.num_layers):
            x = self.mamba_block(x)
        
        # # Compute pairwise squared Euclidean distance matrix D using Ψ
        # Psi_dot = torch.matmul(self.Psi, self.Psi.T)  # ΨΨ^T (N x N)

        # # Compute D (pairwise squared Euclidean distances)
        # D = torch.diag(Psi_dot)[:, None] + torch.diag(Psi_dot) - 2 * Psi_dot  # (N x N)

        # # Compute the adjacency matrix A_G using the Gaussian kernel
        # A_G = torch.exp(-self.psi * D)  # Apply Gaussian kernel with scaling factor ψ

        # # Normalize the adjacency matrix using Softmax (row-wise)
        # A_G = F.softmax(A_G, dim=1)  # Row-wise softmax to normalize

        # # Create the edge_index from the adjacency matrix
        # edge_index = A_G.nonzero(as_tuple=True)  # Returns (row, column) indices where edges exist
        # self.edge_index = torch.stack(edge_index, dim=0)  # (2, num_edges) tensor

        # # Apply the Graph Convolutional Network (GCN) layer
        # x = self.gcn(x, self.edge_index.to(device))
        
        # # Flatten the output to prepare for the final prediction layer
        # x = self.flatten(x)

        x = self.agc_block(x)
        
        # Apply dropout regularization
        x = self.dropout(x)
        
        # Final projection to the prediction length (pred_len)
        x_out = self.projection(x)
        
        return x_out