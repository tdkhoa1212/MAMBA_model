import torch.nn as nn
from mamba.mamba import Mamba
import torch.nn.functional as F
from Models.Blocks.BidirectionalMamba import BidirectionalMambaBlock
from Models.Blocks.AdaptiveGraphConvolution import AdaptiveGraphConvolutionBlock

class GRAPH_MAMBA(nn.Module):
    def __init__(self, configs):
        super(GRAPH_MAMBA, self).__init__()
        self.configs = configs
        self.mamba_block = BidirectionalMambaBlock(configs)
        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            cheb_k=configs.cheb_k, 
            feature_dim=configs.feature_dim
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=configs.node_num)
        self.Flatten = nn.Flatten(start_dim=1)
        
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(configs.linear_depth, configs.pred_len, bias=True)

    def forward(self, input):
        x1=input
        for i in range(self.configs.num_layers):
            x1 = self.mamba_block(x1) 
            x1 = self.dropout(x1)
            x1 = self.relu(x1)

        x1 = self.agc_block(x1) 
        x1 = self.layer_norm(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)

        x = self.projection(x1)
        
        return x