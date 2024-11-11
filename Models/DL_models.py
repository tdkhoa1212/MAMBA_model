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
        self.mamba_block = BidirectionalMambaBlock(pred_len=configs.pred_len, 
                                                       d_model=configs.d_model, 
                                                       d_state=configs.d_state, 
                                                       seq_len=configs.seq_len, 
                                                       num_layers=configs.num_layers,
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

    def forward(self, input_):

        x=input_
        for i in range(self.configs.num_layers):
            x = self.mamba_block(x) 

        x = self.agc_block(x) 
        x = self.dropout(x)
        x_out = self.projection(x)
        
        return x_out