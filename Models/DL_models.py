import torch.nn as nn
# from mamba.mamba import Mamba
import torch.nn.functional as F
from Models.Blocks.BidirectionalMamba import BidirectionalMambaBlock
from Models.Blocks.AdaptiveGraphConvolution import AdaptiveGraphConvolutionBlock
import torch

class GRAPH_MAMBA(nn.Module):
    def __init__(self, configs):
        super(GRAPH_MAMBA, self).__init__()
        self.configs = configs

        self.mamba_block = BidirectionalMambaBlock(configs.pred_len, 
                                                       configs.d_model, 
                                                       configs.d_state, 
                                                       configs.seq_len, 
                                                       configs.num_layers,
                                                       expand=configs.expand,
                                                        hidden_dimention=configs.hidden_dimention)

        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            cheb_k=configs.cheb_k, 
            feature_dim=configs.feature_dim,
        )

        self.l2_lambda = 1e-4
        # self.dropout = nn.Dropout(p=0.1)
        self.projection = nn.Linear(configs.linear_depth, configs.pred_len, bias=True)

    def forward(self, input):

        x2=input
        for i in range(self.configs.num_layers):
            x2 = self.mamba_block(x2) 

        embed = self.agc_block(x2) 

        x_out = self.projection(embed)
        x_out_binary = torch.sigmoid(self.projection(embed))
        
        return x_out, x_out_binary