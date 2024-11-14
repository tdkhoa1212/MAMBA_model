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
        
        self.mamba_block = BidirectionalMambaBlock(
            pred_len=configs.pred_len, 
            d_model=configs.d_model, 
            d_state=configs.d_state, 
            seq_len=configs.seq_len, 
            expand=configs.expand,
            hidden_dimention=configs.hidden_dimention
        )

        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            cheb_k=configs.cheb_k, 
            feature_dim=configs.feature_dim,
        )

        # Regularization parameters
        self.l2_lambda = 1e-4
        self.dropout = nn.Dropout(p=0.2)

        self.projection = nn.Linear(configs.linear_depth, configs.pred_len, bias=True)

    def forward(self, input_):
        """
        Forward pass through the model:
        1. Apply the Mamba block to process the input sequence.
        2. Apply the GCN layer to capture graph-based dependencies.
        3. Apply the final linear projection to get the prediction.
        """
        x = input_
        for i in range(self.configs.num_layers):
            x = self.mamba_block(x)
        
        x = self.agc_block(x)
        x = self.dropout(x)
        x_out = self.projection(x)
        
        return x_out